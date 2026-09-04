package airflow

import (
	"archive/zip"
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime/debug"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/glusman-team/dakp/go/internal/blake3store"
	"github.com/glusman-team/dakp/go/internal/faers"
	"golang.org/x/sync/errgroup"
)

// faersCaseColumns is the rich interim case schema (mirrors faers_ascii._CASE_COLUMNS). The native
// streaming task preserves the full rich case row, including NDA raw value, role/sequence metadata,
// source file, and source_record_id, so downstream assertion shapers can trace exact FAERS records.
var faersCaseColumns = []string{
	"quarter", "primaryid", "caseid", "source", "occp_cod", "reporter_country",
	"drugname", "ingredient", "nda", "nda_raw", "role_cod", "drug_seq", "indi_drug_seq",
	"indication", "effects", "source_file", "source_record_id",
}

var faersWarningsColumns = []string{"quarter", "family", "code", "message", "count"}

// faersEvent prefixes every stat line the FAERS extractor emits (one stat per line).
const faersEvent = "extract_faers"

// faersTaskOperation is the task-level operation-index key for the already-done skip (the
// per-artifact manifests keep their own operation names; see opindex.go).
const faersTaskOperation = "extract_faers"

// ExtractFAERS is the native Go implementation of the DAG's extract_faers stage. It mirrors
// faers_ascii.FAERSASCIIExtractor semantics (DELETE-filtered, cross-quarter-deduped case join)
// but runs as a **bounded-memory streaming pipeline** (plans/fix-faers-memory.md): quarters are
// processed one at a time, most-recent-first — each quarter's family files stream through
// faers.ParseStream (never buffered whole), the case join + cross-quarter caseid dedup run per
// quarter, kept rows spill to sorted scratch run files, and a k-way merge emits the global
// cases.parquet FIRST (so the observed-uses shaper resolves it) plus the public
// faers_cases.tsv in a single pass. Peak memory is one quarter's tables + the dedup-key map,
// NOT the whole corpus. Returns the five ArtifactRefs in the same order as before.
func ExtractFAERS(ctx context.Context, cfg Config, inputs []ArtifactRef) ([]ArtifactRef, error) {
	if cfg.MemoryBudgetGB > 0 {
		debug.SetMemoryLimit(int64(cfg.MemoryBudgetGB) << 30)
		Stat(ctx, faersEvent, "memory_budget_gb", cfg.MemoryBudgetGB)
	}

	store := Store{Workdir: cfg.Workdir}
	upstreamIDs := upstreamInputIDs(inputs)
	if !cfg.Force {
		// Already-done skip: the same upstream artifacts were extracted before.
		if cached := store.FindByOperation(faersTaskOperation, upstreamIDs); cached != nil {
			Stat(ctx, faersEvent, "skipped_already_done", len(cached))
			return cached, nil
		}
	}
	faersDir := filepath.Join(store.InterimDir(), "faers")
	if err := os.MkdirAll(faersDir, 0o755); err != nil {
		return nil, err
	}

	scratch, err := os.MkdirTemp("", "dakp-faers-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(scratch)
	inDir := filepath.Join(scratch, "in")
	if err := StageInputs(inputs, inDir); err != nil {
		return nil, err
	}
	Stat(ctx, faersEvent, "staged_inputs", len(inputs))
	for i, ref := range inputs {
		StatDebug(ctx, faersEvent, fmt.Sprintf("input[%d].uri", i), ref.URI)
		StatDebug(ctx, faersEvent, fmt.Sprintf("input[%d].blake3", i), ref.Blake3)
	}

	quarters, err := inventoryFAERSQuarters(inDir)
	if err != nil {
		return nil, err
	}
	if len(quarters) == 0 {
		return nil, fmt.Errorf("extract_faers: no FAERS ASCII .txt sources staged in %s", inDir)
	}
	quarterList := make([]string, 0, len(quarters))
	for q := range quarters {
		quarterList = append(quarterList, q)
	}
	// Most-recent-first: the streaming dedup relies on newer quarters arriving first, and it
	// mirrors sorted(by_quarter_family, reverse=True) in the Python reference.
	sort.Slice(quarterList, func(i, j int) bool { return quarterList[i] > quarterList[j] })
	Stat(ctx, faersEvent, "quarters_discovered", len(quarterList))
	for _, q := range quarterList {
		StatDebug(ctx, faersEvent, "quarter", q)
	}

	warn := &faers.Warnings{}
	deduper := faers.NewDeduper()
	runsDir := filepath.Join(scratch, "runs")
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		return nil, err
	}

	var (
		runs         []string
		dedupAudit   []faers.DedupAuditRow
		deleteTables []*faers.Table
	)
	for _, q := range quarterList {
		quarterStart := time.Now()
		families, err := parseFAERSQuarter(ctx, q, quarters[q], warn)
		if err != nil {
			return nil, fmt.Errorf("extract_faers: quarter %s: %w", q, err)
		}
		parsedRows := 0
		for family, tbl := range families {
			parsedRows += len(tbl.Rows)
			StatDebug(ctx, faersEvent, q+"."+family+"_rows", len(tbl.Rows))
		}
		if del := families["DELETE"]; del != nil && len(del.Rows) > 0 {
			deleteTables = append(deleteTables, del)
		}
		cases := faers.BuildQuarterCases(families, q, faers.DeletedPrimaryIDs(families["DELETE"]), warn)
		kept, superseded, newKeys := deduper.Split(q, cases)
		deduper.Commit(q, newKeys)
		dedupAudit = append(dedupAudit, superseded...)
		if len(kept) > 0 {
			run := filepath.Join(runsDir, fmt.Sprintf("cases_%s.parquet", q))
			if err := writeFAERSRun(run, kept); err != nil {
				return nil, fmt.Errorf("extract_faers: write run %s: %w", q, err)
			}
			runs = append(runs, run)
		}
		Stat(ctx, faersEvent, q+".parsed_rows", parsedRows)
		Stat(ctx, faersEvent, q+".cases", len(cases))
		Stat(ctx, faersEvent, q+".kept", len(kept))
		Stat(ctx, faersEvent, q+".superseded", len(superseded))
		Stat(ctx, faersEvent, q+".elapsed_s", fmt.Sprintf("%.3f", time.Since(quarterStart).Seconds()))
	}
	faers.SortDedupAudit(dedupAudit)
	Stat(ctx, faersEvent, "warnings_total", int64(warn.Total()))

	inputIDs := make([]string, len(inputs))
	for i, ref := range inputs {
		inputIDs[i] = ref.Blake3
	}
	warningsTotal := int64(warn.Total())

	// 1+2. Global cases.parquet and public faers_cases.tsv, emitted in ONE merged pass.
	casesPath := filepath.Join(faersDir, "cases.parquet")
	tsvPath := filepath.Join(faersDir, "faers_cases.tsv")
	mergeStart := time.Now()
	casesN, err := mergeFAERSOutputs(runs, casesPath, tsvPath)
	if err != nil {
		return nil, fmt.Errorf("extract_faers: merge cases: %w", err)
	}
	Stat(ctx, faersEvent, "merged_rows", casesN)
	Stat(ctx, faersEvent, "merge_elapsed_s", fmt.Sprintf("%.3f", time.Since(mergeStart).Seconds()))
	// 1+2 registered as one batch (both files already exist; hashes run concurrently, refs
	// and stat lines keep their order).
	pairRefs, err := store.RegisterMany(ctx, []RegisterInput{
		{
			Path: casesPath, MediaType: ParquetMediaType, Rows: int64(casesN),
			SchemaFingerprint: SchemaFingerprint(faersCaseColumns), Inputs: inputIDs,
			Warnings: warningsTotal, Operation: "extract_faers_cases",
		},
		{
			Path: tsvPath, MediaType: TSVMediaType, Rows: int64(casesN),
			SchemaFingerprint: SchemaFingerprint(faers.CasesTSVColumns), Inputs: inputIDs,
			Warnings: warningsTotal, Operation: "emit_faers_cases_tsv",
		},
	})
	if err != nil {
		return nil, err
	}
	casesRef, tsvRef := pairRefs[0], pairRefs[1]
	StatOutput(ctx, faersEvent, "cases.parquet", casesRef)
	StatOutput(ctx, faersEvent, "faers_cases.tsv", tsvRef)

	// 3+4+5. delete_audit / dedup_audit / warnings parquets: written, then registered as
	// one batch (hashes run concurrently; refs and stat lines keep their order).
	deleteRows := make([][]string, 0)
	for _, r := range faers.BuildDeleteAudit(deleteTables) {
		deleteRows = append(deleteRows, []string{r.Quarter, r.PrimaryID, r.CaseID, r.SourceFile, r.SourceRecordID})
	}
	deleteIn, err := writeFAERSAuditParquet(faersDir, "delete_audit", faers.DeleteAuditColumns, deleteRows, inputIDs, warningsTotal)
	if err != nil {
		return nil, err
	}
	dedupRows := make([][]string, 0, len(dedupAudit))
	for _, r := range dedupAudit {
		dedupRows = append(dedupRows, []string{r.Quarter, r.PrimaryID, r.CaseID, r.DedupKey, r.WinningQuarter, r.SourceFile})
	}
	dedupIn, err := writeFAERSAuditParquet(faersDir, "dedup_audit", faers.DedupAuditColumns, dedupRows, inputIDs, warningsTotal)
	if err != nil {
		return nil, err
	}
	// warnings.parquet is empty: the Go task records warnings in its log stream, not rows.
	warningsPath := filepath.Join(faersDir, "warnings.parquet")
	if _, err := WriteStringParquet(warningsPath, faersWarningsColumns, nil); err != nil {
		return nil, fmt.Errorf("extract_faers: write warnings.parquet: %w", err)
	}
	auditRefs, err := store.RegisterMany(ctx, []RegisterInput{
		deleteIn,
		dedupIn,
		{
			Path: warningsPath, MediaType: ParquetMediaType, Rows: 0,
			SchemaFingerprint: SchemaFingerprint(faersWarningsColumns), Inputs: inputIDs,
			Warnings: warningsTotal, Operation: "extract_faers_warnings",
		},
	})
	if err != nil {
		return nil, err
	}
	deleteRef, dedupRef, warningsRef := auditRefs[0], auditRefs[1], auditRefs[2]
	StatOutput(ctx, faersEvent, "delete_audit.parquet", deleteRef)
	StatOutput(ctx, faersEvent, "dedup_audit.parquet", dedupRef)
	StatOutput(ctx, faersEvent, "warnings.parquet", warningsRef)

	refs := []ArtifactRef{casesRef, tsvRef, deleteRef, dedupRef, warningsRef}
	Stat(ctx, faersEvent, "output_refs", len(refs))
	store.RecordOperation(faersTaskOperation, upstreamIDs, refs)
	return refs, nil
}

// mergeFAERSOutputs k-way-merges the per-quarter sorted rich runs and streams the merged rows into
// cases.parquet (full CaseColumns via faersCaseRow17) and faers_cases.tsv (public projection)
// simultaneously. Returns the merged row count.
func mergeFAERSOutputs(runs []string, casesPath, tsvPath string) (int, error) {
	pw, err := NewStringParquetWriter(casesPath, faersCaseColumns)
	if err != nil {
		return 0, err
	}
	tsvFile, err := os.Create(tsvPath)
	if err != nil {
		pw.Close()
		return 0, err
	}
	tsv := bufio.NewWriterSize(tsvFile, 1<<20)
	if _, err := tsv.WriteString(strings.Join(faers.CasesTSVColumns, "\t") + "\n"); err != nil {
		pw.Close()
		tsvFile.Close()
		return 0, err
	}
	n := 0
	merr := mergeFAERSRuns(runs, func(row []string) error {
		if err := pw.Append(faersCaseRow17(row)); err != nil {
			return err
		}
		if err := writeFAERSRunTSVRow(tsv, row); err != nil {
			return err
		}
		n++
		return nil
	})
	if merr != nil {
		pw.Close()
		tsvFile.Close()
		return 0, merr
	}
	if err := tsv.Flush(); err != nil {
		pw.Close()
		tsvFile.Close()
		return 0, err
	}
	if err := tsvFile.Close(); err != nil {
		pw.Close()
		return 0, err
	}
	if _, err := pw.Close(); err != nil {
		return 0, err
	}
	return n, nil
}

// writeFAERSAuditParquet writes one audit parquet and returns its RegisterInput; the
// caller batches the registration via RegisterMany.
func writeFAERSAuditParquet(dir, name string, columns []string, rows [][]string, inputIDs []string, warnings int64) (RegisterInput, error) {
	path := filepath.Join(dir, name+".parquet")
	n, err := WriteStringParquet(path, columns, rows)
	if err != nil {
		return RegisterInput{}, fmt.Errorf("extract_faers: write %s: %w", name, err)
	}
	return RegisterInput{
		Path: path, MediaType: ParquetMediaType, Rows: int64(n),
		SchemaFingerprint: SchemaFingerprint(columns), Inputs: inputIDs,
		Warnings: warnings, Operation: "extract_faers_" + name,
	}, nil
}

// --- lazy source inventory (no bulk content reads) -------------------------------

// faersSourceHandle is one logical FAERS ASCII file opened on demand — loose .txt or a zip
// member. Content is never buffered up front: ParseStream hashes + parses it in one pass.
type faersSourceHandle struct {
	Family string
	Name   string // basename used in provenance (source_file)
	Open   func() (io.ReadCloser, error)
}

// zipSession tracks an open quarterly zip shared by the handles of its members; it is closed
// when the last quarter drawing from it has been parsed.
type zipSession struct {
	rc   *zip.ReadCloser
	refs int
}

func (z *zipSession) release() error {
	z.refs--
	if z.refs <= 0 {
		return z.rc.Close()
	}
	return nil
}

// faersQuarterSources is one quarter's lazy family handles plus the zips to release after it.
type faersQuarterSources struct {
	families map[string]faersSourceHandle
	zips     []*zipSession
}

func (qs *faersQuarterSources) closeZips() {
	for _, z := range qs.zips {
		z.release()
	}
}

// inventoryFAERSQuarters scans inDir (recursively) for FAERS ASCII sources — loose .txt files
// and .txt members inside quarterly ASCII .zip archives (the shape the FAERS fetcher
// downloads) — and returns quarter -> lazy family handles. Nothing is read into memory here;
// each handle streams its content when opened. Deterministic: paths walk in sorted order.
func inventoryFAERSQuarters(inDir string) (map[string]*faersQuarterSources, error) {
	var txtPaths, zipPaths []string
	err := filepath.WalkDir(inDir, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			return nil
		}
		switch {
		case strings.EqualFold(filepath.Ext(path), ".txt"):
			txtPaths = append(txtPaths, path)
		case strings.EqualFold(filepath.Ext(path), ".zip"):
			zipPaths = append(zipPaths, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(txtPaths)
	sort.Strings(zipPaths)

	out := map[string]*faersQuarterSources{}
	get := func(q string) *faersQuarterSources {
		if out[q] == nil {
			out[q] = &faersQuarterSources{families: map[string]faersSourceHandle{}}
		}
		return out[q]
	}

	for _, p := range txtPaths {
		name := filepath.Base(p)
		family, quarter := faers.FamilyAndQuarter(name)
		if family == "" || quarter == "" {
			continue // not a FAERS ASCII file
		}
		path := p
		get(quarter).families[family] = faersSourceHandle{
			Family: family,
			Name:   name,
			Open:   func() (io.ReadCloser, error) { return os.Open(path) },
		}
	}

	for _, zp := range zipPaths {
		reader, err := zip.OpenReader(zp)
		if err != nil {
			return nil, err
		}
		sess := &zipSession{rc: reader}
		zipQuarters := map[string]bool{}
		for _, file := range reader.File {
			if file.FileInfo().IsDir() || !strings.EqualFold(filepath.Ext(file.Name), ".txt") {
				continue
			}
			name := filepath.Base(file.Name)
			family, quarter := faers.FamilyAndQuarter(name)
			if family == "" || quarter == "" {
				continue
			}
			mem := file
			qs := get(quarter)
			qs.families[family] = faersSourceHandle{Family: family, Name: name, Open: mem.Open}
			if !zipQuarters[quarter] {
				zipQuarters[quarter] = true
				sess.refs++
				qs.zips = append(qs.zips, sess)
			}
		}
		if sess.refs == 0 { // zip held no FAERS members
			if err := reader.Close(); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

// parseFAERSQuarter streams one quarter's family files into normalized Tables concurrently
// (bounded by the family count — at most six files, never the whole corpus at once). The
// quarter's zips are released before returning.
func parseFAERSQuarter(ctx context.Context, quarter string, qs *faersQuarterSources, warn *faers.Warnings) (map[string]*faers.Table, error) {
	defer qs.closeZips()
	var (
		mu       sync.Mutex
		families = map[string]*faers.Table{}
	)
	g, gctx := errgroup.WithContext(ctx)
	for family, h := range qs.families {
		family, h := family, h
		g.Go(func() error {
			if err := gctx.Err(); err != nil {
				return err
			}
			rc, err := h.Open()
			if err != nil {
				return err
			}
			defer rc.Close()
			tbl := faers.ParseStream(rc, quarter, family, h.Name, warn)
			if tbl == nil || len(tbl.Rows) == 0 {
				return nil
			}
			mu.Lock()
			families[family] = tbl
			mu.Unlock()
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}
	return families, nil
}

// --- batch source loading (kept for the parity tests + dakp-worker-style checks) ---------

// loadFAERSSources collects the FAERS ASCII sources from inDir: loose .txt files (pre-unpacked
// quarters) and .txt members inside quarterly ASCII .zip archives (the shape the FAERS fetcher
// downloads). Each kept source's basename resolves to a FAERS family + quarter; content is hashed
// for the b3-derived source_record_id. Sorted path order for determinism. Mirrors
// faers_ascii._iter_faers_sources (loose .txt or zip members).
//
// NOTE: this batch loader reads whole files into memory and is used ONLY by the parity tests as
// the canonical oracle; the production ExtractFAERS path uses inventoryFAERSQuarters above.
func loadFAERSSources(inDir string) ([]faers.Source, error) {
	var txtPaths, zipPaths []string
	err := filepath.WalkDir(inDir, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			return nil
		}
		switch {
		case strings.EqualFold(filepath.Ext(path), ".txt"):
			txtPaths = append(txtPaths, path)
		case strings.EqualFold(filepath.Ext(path), ".zip"):
			zipPaths = append(zipPaths, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(txtPaths)
	sort.Strings(zipPaths)

	var srcs []faers.Source
	for _, p := range txtPaths {
		src, ok, err := faersSourceFromLoose(p)
		if err != nil {
			return nil, err
		}
		if ok {
			srcs = append(srcs, src)
		}
	}
	for _, zp := range zipPaths {
		zipSrcs, err := faersSourcesFromZip(zp)
		if err != nil {
			return nil, err
		}
		srcs = append(srcs, zipSrcs...)
	}
	return srcs, nil
}

// faersSourceFromLoose loads one loose .txt file as a faers.Source (ok=false if its name does not
// resolve to a FAERS family + quarter).
func faersSourceFromLoose(path string) (faers.Source, bool, error) {
	name := filepath.Base(path)
	family, quarter := faers.FamilyAndQuarter(name)
	if family == "" || quarter == "" {
		return faers.Source{}, false, nil
	}
	content, err := os.ReadFile(path)
	if err != nil {
		return faers.Source{}, false, err
	}
	return faers.Source{
		Quarter: quarter, Family: family, Content: content,
		SourceName: name, SourceB3: blake3store.HashBytes(content),
	}, true, nil
}

// faersSourcesFromZip reads each .txt member of a quarterly ASCII zip directly (no disk unpack),
// keeping members whose basename resolves to a FAERS family + quarter.
func faersSourcesFromZip(zipPath string) ([]faers.Source, error) {
	reader, err := zip.OpenReader(zipPath)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	var srcs []faers.Source
	for _, file := range reader.File {
		if file.FileInfo().IsDir() || !strings.EqualFold(filepath.Ext(file.Name), ".txt") {
			continue
		}
		name := filepath.Base(file.Name)
		family, quarter := faers.FamilyAndQuarter(name)
		if family == "" || quarter == "" {
			continue
		}
		rc, err := file.Open()
		if err != nil {
			return nil, err
		}
		content, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			return nil, err
		}
		srcs = append(srcs, faers.Source{
			Quarter: quarter, Family: family, Content: content,
			SourceName: name, SourceB3: blake3store.HashBytes(content),
		})
	}
	return srcs, nil
}
