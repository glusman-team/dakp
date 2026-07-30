package airflow

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/glusman-team/dakp/go/internal/blake3store"
	"github.com/glusman-team/dakp/go/internal/faers"
)

// faersCaseColumns is the rich interim case schema (mirrors faers_ascii._CASE_COLUMNS). The native
// task reconstructs cases.parquet from the public case projection (as _extract_via_go did), so the
// provenance columns (nda_raw/role_cod/drug_seq/indi_drug_seq/source_file/source_record_id) are
// emitted empty; the public columns carry the data.
var faersCaseColumns = []string{
	"quarter", "primaryid", "caseid", "source", "occp_cod", "reporter_country",
	"drugname", "ingredient", "nda", "nda_raw", "role_cod", "drug_seq", "indi_drug_seq",
	"indication", "effects", "source_file", "source_record_id",
}

var faersWarningsColumns = []string{"quarter", "family", "code", "message", "count"}

// ExtractFAERS is the native Go implementation of the DAG's extract_faers stage. It mirrors
// faers_ascii.FAERSASCIIExtractor._extract_via_go: parse the staged FAERS ASCII sources with
// internal/faers (DELETE-filtered, cross-quarter-deduped case join), then register the global
// cases.parquet FIRST (so the observed-uses shaper resolves it), the public faers_cases.tsv
// handoff, the delete/dedup audit parquets, and an empty warnings parquet. Returns the five
// ArtifactRefs in the same order as the Python extractor.
func ExtractFAERS(ctx context.Context, cfg Config, inputs []ArtifactRef) ([]ArtifactRef, error) {
	store := Store{Workdir: cfg.Workdir}
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

	srcs, err := loadFAERSSources(inDir)
	if err != nil {
		return nil, err
	}
	if len(srcs) == 0 {
		return nil, fmt.Errorf("extract_faers: no FAERS ASCII .txt sources staged in %s", inDir)
	}

	warn := &faers.Warnings{}
	limit := cfg.Threads
	if limit <= 0 {
		limit = 4
	}
	res, err := faers.Extract(ctx, srcs, limit, warn)
	if err != nil {
		return nil, err
	}
	warningsTotal := int64(warn.Total())
	inputIDs := make([]string, len(inputs))
	for i, ref := range inputs {
		inputIDs[i] = ref.Blake3
	}

	// 1. Global cases.parquet (reconstructed from the public projection; provenance columns empty).
	caseRows := make([][]string, len(res.Cases))
	for i, c := range res.Cases {
		caseRows[i] = []string{
			c.Quarter, c.PrimaryID, c.CaseID, c.Source, c.OccpCod, c.ReporterCountry,
			c.Drugname, c.Ingredient, c.Nda, "", "", "", "", c.Indication, c.Effects, "", "",
		}
	}
	casesPath := filepath.Join(faersDir, "cases.parquet")
	casesN, err := WriteStringParquet(casesPath, faersCaseColumns, caseRows)
	if err != nil {
		return nil, fmt.Errorf("extract_faers: write cases.parquet: %w", err)
	}
	casesRef, err := store.Register(RegisterInput{
		Path: casesPath, MediaType: ParquetMediaType, Rows: int64(casesN),
		SchemaFingerprint: SchemaFingerprint(faersCaseColumns), Inputs: inputIDs,
		Warnings: warningsTotal, Operation: "extract_faers_cases",
	})
	if err != nil {
		return nil, err
	}

	// 2. Public faers_cases.tsv (Tablassert source-section handoff).
	tsvPath := filepath.Join(faersDir, "faers_cases.tsv")
	if err := writeFAERSCasesTSV(tsvPath, res.Cases); err != nil {
		return nil, fmt.Errorf("extract_faers: write faers_cases.tsv: %w", err)
	}
	tsvRef, err := store.Register(RegisterInput{
		Path: tsvPath, MediaType: TSVMediaType, Rows: int64(len(res.Cases)),
		SchemaFingerprint: SchemaFingerprint(faers.CasesTSVColumns), Inputs: inputIDs,
		Warnings: warningsTotal, Operation: "emit_faers_cases_tsv",
	})
	if err != nil {
		return nil, err
	}

	// 3. delete_audit.parquet.
	deleteRows := make([][]string, len(res.DeleteAudit))
	for i, r := range res.DeleteAudit {
		deleteRows[i] = []string{r.Quarter, r.PrimaryID, r.CaseID, r.SourceFile, r.SourceRecordID}
	}
	deleteRef, err := writeFAERSAuditParquet(store, faersDir, "delete_audit", faers.DeleteAuditColumns, deleteRows, inputIDs, warningsTotal)
	if err != nil {
		return nil, err
	}

	// 4. dedup_audit.parquet.
	dedupRows := make([][]string, len(res.DedupAudit))
	for i, r := range res.DedupAudit {
		dedupRows[i] = []string{r.Quarter, r.PrimaryID, r.CaseID, r.DedupKey, r.WinningQuarter, r.SourceFile}
	}
	dedupRef, err := writeFAERSAuditParquet(store, faersDir, "dedup_audit", faers.DedupAuditColumns, dedupRows, inputIDs, warningsTotal)
	if err != nil {
		return nil, err
	}

	// 5. warnings.parquet (empty; the Go task records warnings in its log stream, not rows).
	warningsPath := filepath.Join(faersDir, "warnings.parquet")
	if _, err := WriteStringParquet(warningsPath, faersWarningsColumns, nil); err != nil {
		return nil, fmt.Errorf("extract_faers: write warnings.parquet: %w", err)
	}
	warningsRef, err := store.Register(RegisterInput{
		Path: warningsPath, MediaType: ParquetMediaType, Rows: 0,
		SchemaFingerprint: SchemaFingerprint(faersWarningsColumns), Inputs: inputIDs,
		Warnings: warningsTotal, Operation: "extract_faers_warnings",
	})
	if err != nil {
		return nil, err
	}

	return []ArtifactRef{casesRef, tsvRef, deleteRef, dedupRef, warningsRef}, nil
}

func writeFAERSAuditParquet(store Store, dir, name string, columns []string, rows [][]string, inputIDs []string, warnings int64) (ArtifactRef, error) {
	path := filepath.Join(dir, name+".parquet")
	n, err := WriteStringParquet(path, columns, rows)
	if err != nil {
		return ArtifactRef{}, fmt.Errorf("extract_faers: write %s: %w", name, err)
	}
	return store.Register(RegisterInput{
		Path: path, MediaType: ParquetMediaType, Rows: int64(n),
		SchemaFingerprint: SchemaFingerprint(columns), Inputs: inputIDs,
		Warnings: warnings, Operation: "extract_faers_" + name,
	})
}

func writeFAERSCasesTSV(path string, cases []faers.Case) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := faers.WriteCasesTSV(f, cases); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// loadFAERSSources walks inDir (recursively) for .txt files, keeps those whose basename resolves
// to a FAERS family + quarter, and loads each as a faers.Source (content-hashed for the b3-derived
// source_record_id). Sorted path order for determinism. Mirrors the CLI's loadFAERSSources.
func loadFAERSSources(inDir string) ([]faers.Source, error) {
	var paths []string
	err := filepath.WalkDir(inDir, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			return nil
		}
		if strings.EqualFold(filepath.Ext(path), ".txt") {
			paths = append(paths, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(paths)

	var srcs []faers.Source
	for _, p := range paths {
		family, quarter := faers.FamilyAndQuarter(filepath.Base(p))
		if family == "" || quarter == "" {
			continue
		}
		content, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}
		srcs = append(srcs, faers.Source{
			Quarter:    quarter,
			Family:     family,
			Content:    content,
			SourceName: filepath.Base(p),
			SourceB3:   blake3store.HashBytes(content),
		})
	}
	return srcs, nil
}
