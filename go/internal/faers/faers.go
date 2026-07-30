// Package faers implements the FAERS ASCII extractor: it parses each quarter's
// `$`-delimited FAERS ASCII files into normalized per-family tables (DEMO, DRUG, INDI,
// REAC, RPSR, DELETE) and builds the per-quarter case-level join honoring DELETEd
// primaryids and cross-quarter caseid dedup (most-recent-wins).
//
// It is a faithful Go port of src/dakp_pipeline/extract/faers_ascii.py (Milestone 3):
// the normalized-table column layout, the per-row source_record_id derivation, the
// case-join semantics, the dedup/DELETE audits, and the uncompressed public TSV contract
// (schemas.FAERS_CASES_COLUMNS) all match the Python reference. The join semantics are
// themselves ported from ref/legacy/FAERS/bin/listCases.pl (case rows are driven by INDI
// joined to DRUG on (primaryid, drug_seq == indi_drug_seq), with DEMO reporter metadata,
// RPSR source, and REAC reactions `$`-joined per case).
//
// Parsing robustness for real FDA ASCII mirrors the Python:
//   - `$` delimiter with a trailing `$` on every line (trailing empty column, dropped);
//   - CRLF line endings (the `\r` is trimmed);
//   - UPPERCASE headers normalized to lowercase;
//   - primaryid / legacy isr column resolution (pre-2014 FAERS used isr).
//
// Files are parsed with bufio streaming line reads (FAERS files are huge) and, at the
// Extract entrypoint, concurrently with bounded parallelism (errgroup SetLimit).
package faers

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"

	"github.com/glusman-team/dakp/go/internal/blake3store"
	"golang.org/x/sync/errgroup"
)

// Delimiter separates FAERS ASCII fields.
const Delimiter = "$"

// Families are the logical FAERS ASCII table families, in canonical match order (mirrors
// faers_ascii._FAMILIES). A filename's family is the first family that prefixes its
// uppercased basename (e.g. "DEMO24Q3.txt" -> DEMO).
var Families = []string{"DEMO", "DRUG", "INDI", "REAC", "RPSR", "DELETE"}

// CasesTSVColumns is the uncompressed public TSV contract for faers_cases.tsv (mirrors
// schemas.FAERS_CASES_COLUMNS — the Tablassert source-section handoff). It is a projection
// of the richer CaseColumns.
var CasesTSVColumns = []string{
	"quarter", "primaryid", "caseid", "source", "occp_cod", "reporter_country",
	"drugname", "ingredient", "nda", "indication", "effects",
}

// CaseColumns is the full rich per-case schema (mirrors faers_ascii._CASE_COLUMNS), in
// order. The public TSV projects a subset (CasesTSVColumns); the extra columns (nda_raw,
// role_cod, drug_seq, indi_drug_seq, source_file, source_record_id) are preserved on Case
// for downstream shaping and provenance.
var CaseColumns = []string{
	"quarter", "primaryid", "caseid", "source", "occp_cod", "reporter_country",
	"drugname", "ingredient", "nda", "nda_raw", "role_cod", "drug_seq",
	"indi_drug_seq", "indication", "effects", "source_file", "source_record_id",
}

// DeleteAuditColumns / DedupAuditColumns are the audit-table column orders (mirror
// faers_ascii._DELETE_AUDIT_COLUMNS / _DEDUP_AUDIT_COLUMNS).
var DeleteAuditColumns = []string{"quarter", "primaryid", "caseid", "source_file", "source_record_id"}

var DedupAuditColumns = []string{"quarter", "primaryid", "caseid", "dedup_key", "winning_quarter", "source_file"}

// caseSortKey orders case rows deterministically (mirrors faers_ascii._CASE_SORT_KEY).
// auditSortKey orders audit rows (mirrors _AUDIT_SORT_KEY).
var (
	caseSortKey  = []string{"primaryid", "drug_seq", "indication"}
	auditSortKey = []string{"quarter", "primaryid"}
)

// Source is one logical FAERS ASCII file (a loose .txt; zip members are unpacked to this
// shape by callers). It mirrors faers_ascii._FaersSource.
type Source struct {
	Quarter    string // e.g. "24Q3"
	Family     string // one of Families (uppercase)
	Content    []byte // raw file bytes
	SourceName string // basename used in provenance (source_file)
	SourceB3   string // b3:<hex> content hash of Content
}

// Table is one normalized per-family table: ordered column names (provenance first) and
// rows aligned to Columns. Every value is a trimmed string ("" for missing), matching the
// Python all-Utf8 + strip_chars parse.
type Table struct {
	Quarter string
	Family  string
	Columns []string
	Rows    [][]string
}

// Case is one row of the per-quarter case join (the full CaseColumns schema).
type Case struct {
	Quarter         string
	PrimaryID       string
	CaseID          string
	Source          string // reporter source (rpsr_cod)
	OccpCod         string
	ReporterCountry string
	Drugname        string
	Ingredient      string // prod_ai
	Nda             string // normalized: digits only, leading zeroes stripped
	NdaRaw          string // nda_num verbatim (leading zeroes preserved)
	RoleCod         string
	DrugSeq         string
	IndiDrugSeq     string
	Indication      string // indi_pt
	Effects         string // reactions: sorted-unique pt values, "$"-joined
	SourceFile      string // DRUG source file (provenance)
	SourceRecordID  string // quarter:primaryid:drug_seq:indication
}

// DeleteAuditRow / DedupAuditRow are the audit-table rows.
type DeleteAuditRow struct {
	Quarter        string
	PrimaryID      string
	CaseID         string
	SourceFile     string
	SourceRecordID string
}

type DedupAuditRow struct {
	Quarter        string
	PrimaryID      string
	CaseID         string
	DedupKey       string
	WinningQuarter string
	SourceFile     string
}

// Warning is one parse/extract warning (mirrors faers_ascii._Warning).
type Warning struct {
	Quarter string
	Family  string
	Code    string
	Message string
	Count   int
}

// Warnings accumulates warnings; it is safe for concurrent use (Extract parses files in
// parallel). It mirrors faers_ascii._Warnings.
type Warnings struct {
	mu    sync.Mutex
	items []Warning
}

// Add records a warning with count 1.
func (w *Warnings) Add(quarter, family, code, message string) {
	w.AddCount(quarter, family, code, message, 1)
}

// AddCount records a warning with an explicit count.
func (w *Warnings) AddCount(quarter, family, code, message string, count int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.items = append(w.items, Warning{Quarter: quarter, Family: family, Code: code, Message: message, Count: count})
}

// Total returns the sum of all warning counts.
func (w *Warnings) Total() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	n := 0
	for _, it := range w.items {
		n += it.Count
	}
	return n
}

// Items returns a copy of the recorded warnings.
func (w *Warnings) Items() []Warning {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := make([]Warning, len(w.items))
	copy(out, w.items)
	return out
}

// Has reports whether any warning carries the given code.
func (w *Warnings) Has(code string) bool {
	for _, it := range w.Items() {
		if it.Code == code {
			return true
		}
	}
	return false
}

// quarterInNameRe extracts the two-digit year + quarter from a FAERS filename (mirrors
// faers_ascii._QUARTER_IN_NAME_RE), e.g. "DEMO24Q3.txt" -> ("24","3"), "ascii/DEMO2018q2"
// is NOT matched (needs exactly two year digits; the fetcher canonicalizes full years).
var quarterInNameRe = regexp.MustCompile(`(?i)(\d{2})q(\d)`)

// FamilyAndQuarter derives (family, quarter) from a FAERS filename (mirrors
// faers_ascii._family_and_quarter). family is "" if no family prefixes the basename;
// quarter is "" if no NNqN pattern is present (returned uppercase, e.g. "24Q3").
func FamilyAndQuarter(name string) (family, quarter string) {
	base := strings.ToUpper(filepath.Base(name))
	for _, fam := range Families {
		if strings.HasPrefix(base, fam) {
			family = fam
			break
		}
	}
	if m := quarterInNameRe.FindStringSubmatch(name); m != nil {
		quarter = strings.ToUpper(m[1] + "Q" + m[2])
	}
	return family, quarter
}

// ParseSource parses one `$`-delimited ASCII source into a normalized Table with
// quarter/source_file/source_record_id provenance prepended (mirrors
// faers_ascii._parse_source). It returns nil (after recording a warning) if the source has
// no usable header/rows or no primaryid/isr column. Parsing streams line-by-line via bufio.
func ParseSource(src Source, warn *Warnings) *Table {
	if len(bytes.TrimSpace(src.Content)) == 0 {
		warn.Add(src.Quarter, src.Family, "empty_file", "no bytes")
		return nil
	}
	r := bufio.NewReaderSize(bytes.NewReader(src.Content), 1<<20)

	header, err := readLine(r)
	if err != nil { // EOF (or read error) before any header line
		warn.Add(src.Quarter, src.Family, "empty_file", "no data rows")
		return nil
	}

	// Split the header on "$"; the trailing "$" yields a trailing empty field. Keep only
	// columns whose (trimmed) header name is non-empty, recording their source positions.
	rawCols := strings.Split(header, Delimiter)
	var names []string
	var pos []int
	for i, c := range rawCols {
		name := strings.ToLower(strings.TrimSpace(c))
		if name == "" {
			continue
		}
		names = append(names, name)
		pos = append(pos, i)
	}

	// Resolve the case identifier column: primaryid, or legacy isr renamed to primaryid.
	if !containsString(names, "primaryid") {
		if containsString(names, "isr") {
			for i, n := range names {
				if n == "isr" {
					names[i] = "primaryid"
				}
			}
		} else {
			warn.Add(src.Quarter, src.Family, "missing_primaryid", "no primaryid/isr column")
			return nil
		}
	}

	ncols := len(rawCols)
	var rows [][]string
	for {
		line, rerr := readLine(r)
		if rerr == io.EOF && line == "" {
			break
		}
		if rerr != nil && rerr != io.EOF {
			warn.Add(src.Quarter, src.Family, "parse_error", rerr.Error())
			return nil
		}
		fields := strings.Split(line, Delimiter)
		if len(fields) > ncols { // truncate_ragged_lines: cap at the header width
			fields = fields[:ncols]
		}
		row := make([]string, len(pos))
		for j, p := range pos { // shorter rows leave "" (ragged-line padding)
			if p < len(fields) {
				row[j] = strings.TrimSpace(fields[p])
			}
		}
		rows = append(rows, row)
	}
	if len(rows) == 0 {
		warn.Add(src.Quarter, src.Family, "empty_file", "no data rows")
		return nil
	}

	short := b3Short(src.SourceB3)
	cols := make([]string, 0, 3+len(names))
	cols = append(cols, "quarter", "source_file", "source_record_id")
	cols = append(cols, names...)

	out := make([][]string, 0, len(rows))
	for _, row := range rows {
		full := make([]string, 0, len(cols))
		full = append(full, src.Quarter, src.SourceName, sourceRecordID(short, src.Family, row, names))
		full = append(full, row...)
		out = append(out, full)
	}
	return &Table{Quarter: src.Quarter, Family: src.Family, Columns: cols, Rows: out}
}

// readLine reads one line, stripping the trailing CRLF/LF. On EOF with no remaining data it
// returns ("", io.EOF); a final unterminated line is returned with io.EOF and non-empty s.
func readLine(r *bufio.Reader) (string, error) {
	line, err := r.ReadString('\n')
	if err != nil && line == "" {
		return "", err
	}
	return strings.TrimRight(line, "\r\n"), nil
}

// sourceRecordID derives the deterministic per-row id for a normalized table (mirrors
// faers_ascii._source_record_id): <source-hash-prefix>:<primaryid>[:<seq|pt>], where the
// prefix is the first 12 hex chars of the source file's b3 digest. The suffix parts are
// family-specific (DRUG: drug_seq; INDI: indi_drug_seq, indi_pt; REAC: pt).
func sourceRecordID(short, family string, row []string, names []string) string {
	get := func(name string) string {
		i := indexOf(names, name)
		if i < 0 || i >= len(row) {
			return ""
		}
		return row[i]
	}
	parts := []string{short, get("primaryid")}
	switch family {
	case "DRUG":
		parts = append(parts, get("drug_seq"))
	case "INDI":
		parts = append(parts, get("indi_drug_seq"), get("indi_pt"))
	case "REAC":
		parts = append(parts, get("pt"))
	}
	return strings.Join(parts, ":")
}

// b3Short returns the first 12 hex chars of a b3:<hex> id (mirrors
// source_b3.split(":", 1)[1][:12]).
func b3Short(id string) string {
	s := strings.TrimPrefix(id, blake3store.IDPrefix)
	if len(s) > 12 {
		return s[:12]
	}
	return s
}

// DeletedPrimaryIDs returns the primaryids marked deleted for a quarter (mirrors
// faers_ascii._deleted_primaryids / listCases.pl readDELETE).
func DeletedPrimaryIDs(del *Table) map[string]bool {
	out := map[string]bool{}
	if del == nil || len(del.Rows) == 0 {
		return out
	}
	pid := indexOf(del.Columns, "primaryid")
	if pid < 0 {
		return out
	}
	for _, r := range del.Rows {
		if v := cell(r, pid); v != "" {
			out[v] = true
		}
	}
	return out
}

// BuildQuarterCases joins one quarter's normalized tables into case rows: INDI-driven,
// inner-joined to DRUG on (primaryid, drug_seq == indi_drug_seq), left-joined to DEMO
// (reporter metadata), RPSR (source), and REAC (sorted-unique "$"-joined effects), with
// DELETEd primaryids removed from every family (mirrors faers_ascii._build_quarter_cases).
// The result is sorted by caseSortKey and intra-quarter exact-row deduped (legacy
// listCases.pl %seenRow). Returns nil when DRUG or INDI is absent/empty (no cases).
func BuildQuarterCases(families map[string]*Table, quarter string, deleted map[string]bool, warn *Warnings) []Case {
	drug := families["DRUG"]
	indi := families["INDI"]
	if drug == nil || indi == nil || len(drug.Rows) == 0 || len(indi.Rows) == 0 {
		return nil // case rows are indication-driven; no DRUG/INDI => no cases
	}

	// Index DRUG rows by join key (primaryid|drug_seq), preserving file order, skipping
	// DELETEd primaryids.
	dPid := indexOf(drug.Columns, "primaryid")
	dSeq := indexOf(drug.Columns, "drug_seq")
	dName := indexOf(drug.Columns, "drugname")
	dRole := indexOf(drug.Columns, "role_cod")
	dNda := indexOf(drug.Columns, "nda_num")
	dAI := indexOf(drug.Columns, "prod_ai")
	dSrc := indexOf(drug.Columns, "source_file")
	type drugRow struct{ seq, name, role, nda, ai, src string }
	drugByKey := map[string][]drugRow{}
	drugDeleted := 0
	for _, r := range drug.Rows {
		pid := cell(r, dPid)
		if deleted[pid] {
			drugDeleted++
			continue
		}
		seq := cell(r, dSeq)
		key := pid + "|" + seq
		drugByKey[key] = append(drugByKey[key], drugRow{seq, cell(r, dName), cell(r, dRole), cell(r, dNda), cell(r, dAI), cell(r, dSrc)})
	}

	iPid := indexOf(indi.Columns, "primaryid")
	iSeq := indexOf(indi.Columns, "indi_drug_seq")
	iPt := indexOf(indi.Columns, "indi_pt")

	// DEMO reporter metadata by primaryid (left-join source; may multiply on dup keys).
	type demoRow struct{ caseid, occp, country string }
	demoByPid := map[string][]demoRow{}
	if demo := families["DEMO"]; demo != nil && len(demo.Rows) > 0 {
		mPid := indexOf(demo.Columns, "primaryid")
		mCase := indexOf(demo.Columns, "caseid")
		mOccp := indexOf(demo.Columns, "occp_cod")
		mCountry := indexOf(demo.Columns, "reporter_country")
		for _, r := range demo.Rows {
			pid := cell(r, mPid)
			if deleted[pid] {
				continue
			}
			demoByPid[pid] = append(demoByPid[pid], demoRow{cell(r, mCase), cell(r, mOccp), cell(r, mCountry)})
		}
	}

	// RPSR source codes by primaryid.
	rpsrByPid := map[string][]string{}
	if rpsr := families["RPSR"]; rpsr != nil && len(rpsr.Rows) > 0 {
		rPid := indexOf(rpsr.Columns, "primaryid")
		rCod := indexOf(rpsr.Columns, "rpsr_cod")
		for _, r := range rpsr.Rows {
			pid := cell(r, rPid)
			if deleted[pid] {
				continue
			}
			rpsrByPid[pid] = append(rpsrByPid[pid], cell(r, rCod))
		}
	}

	// REAC effects: per primaryid, sorted-unique non-empty pt values "$"-joined.
	effectsByPid := map[string]string{}
	if reac := families["REAC"]; reac != nil && len(reac.Rows) > 0 {
		ePid := indexOf(reac.Columns, "primaryid")
		ePt := indexOf(reac.Columns, "pt")
		ptsByPid := map[string]map[string]bool{}
		for _, r := range reac.Rows {
			pid := cell(r, ePid)
			if deleted[pid] {
				continue
			}
			pt := cell(r, ePt)
			if pt == "" {
				continue
			}
			if ptsByPid[pid] == nil {
				ptsByPid[pid] = map[string]bool{}
			}
			ptsByPid[pid][pt] = true
		}
		for pid, set := range ptsByPid {
			pts := make([]string, 0, len(set))
			for pt := range set {
				pts = append(pts, pt)
			}
			sort.Strings(pts)
			effectsByPid[pid] = strings.Join(pts, Delimiter)
		}
	}

	emptyDemo := []demoRow{{}}
	emptyRpsr := []string{""}
	var cases []Case
	for _, ir := range indi.Rows {
		pid := cell(ir, iPid)
		if deleted[pid] {
			continue
		}
		iseq := cell(ir, iSeq)
		ipt := cell(ir, iPt)
		drugs := drugByKey[pid+"|"+iseq]
		if len(drugs) == 0 {
			continue // inner join: no matching drug row => no case row
		}
		demos := demoByPid[pid]
		if len(demos) == 0 {
			demos = emptyDemo
		}
		rpsrs := rpsrByPid[pid]
		if len(rpsrs) == 0 {
			rpsrs = emptyRpsr
		}
		effects := effectsByPid[pid] // "" when the case has no reactions
		for _, d := range drugs {
			for _, dm := range demos {
				for _, rp := range rpsrs {
					cases = append(cases, Case{
						Quarter:         quarter,
						PrimaryID:       pid,
						CaseID:          dm.caseid,
						Source:          rp,
						OccpCod:         dm.occp,
						ReporterCountry: dm.country,
						Drugname:        d.name,
						Ingredient:      d.ai,
						Nda:             normalizeNDA(d.nda),
						NdaRaw:          d.nda,
						RoleCod:         d.role,
						DrugSeq:         d.seq,
						IndiDrugSeq:     iseq,
						Indication:      ipt,
						Effects:         effects,
						SourceFile:      d.src,
						SourceRecordID:  strings.Join([]string{quarter, pid, d.seq, ipt}, ":"),
					})
				}
			}
		}
	}

	sortCases(cases)

	// Intra-quarter exact-row dedup on the value subset, keep first in sorted order
	// (legacy listCases.pl %seenRow).
	kept := make([]Case, 0, len(cases))
	seen := make(map[string]bool, len(cases))
	for _, c := range cases {
		k := c.dedupSubsetKey()
		if seen[k] {
			continue
		}
		seen[k] = true
		kept = append(kept, c)
	}

	if len(deleted) > 0 && drugDeleted > 0 {
		warn.AddCount(quarter, "DRUG", "deleted_rows_dropped",
			fmt.Sprintf("%d drug rows dropped (DELETE)", drugDeleted), drugDeleted)
	}
	return kept
}

// ReduceCases concatenates per-quarter cases and applies cross-quarter caseid dedup
// (most-recent-wins): the dedup key is caseid when present, else primaryid (matches
// listCases.pl caseid fallback); only the max-quarter rows per key survive. It returns the
// kept cases (sorted by caseSortKey) and the dedup audit (superseded rows, sorted by
// auditSortKey). Mirrors faers_ascii._reduce_cases.
func ReduceCases(perQuarter [][]Case) (kept []Case, audit []DedupAuditRow) {
	var all []Case
	for _, cs := range perQuarter {
		all = append(all, cs...)
	}
	if len(all) == 0 {
		return nil, nil
	}
	winning := map[string]string{}
	for _, c := range all {
		k := c.dedupKey()
		if q, ok := winning[k]; !ok || c.Quarter > q {
			winning[k] = c.Quarter
		}
	}
	var superseded []Case
	for _, c := range all {
		if c.Quarter == winning[c.dedupKey()] {
			kept = append(kept, c)
		} else {
			superseded = append(superseded, c)
		}
	}
	audit = make([]DedupAuditRow, 0, len(superseded))
	for _, c := range superseded {
		audit = append(audit, DedupAuditRow{
			Quarter:        c.Quarter,
			PrimaryID:      c.PrimaryID,
			CaseID:         c.CaseID,
			DedupKey:       c.dedupKey(),
			WinningQuarter: winning[c.dedupKey()],
			SourceFile:     c.SourceFile,
		})
	}
	sortAudit(audit)
	sortCases(kept)
	return kept, audit
}

// Result is the output of Extract: the global deduped cases, the DELETE and cross-quarter
// dedup audits, the normalized per-quarter/per-family tables, and the processed quarters
// (most-recent-first).
type Result struct {
	Cases       []Case
	DeleteAudit []DeleteAuditRow
	DedupAudit  []DedupAuditRow
	Normalized  map[string]map[string]*Table
	Quarters    []string
}

// Extract parses the sources concurrently (bounded by limit; limit <= 0 means unbounded)
// and runs the full pipeline: group by quarter, build per-quarter cases (DELETE-filtered,
// intra-quarter deduped) most-recent-first, then reduce across quarters (caseid dedup).
// It mirrors FAERSASCIIExtractor.extract (minus parquet/manifest emission, which the CLI
// layer owns).
func Extract(ctx context.Context, srcs []Source, limit int, warn *Warnings) (Result, error) {
	byQF, err := ParseSourcesConcurrent(ctx, srcs, limit, warn)
	if err != nil {
		return Result{}, err
	}
	return ExtractParsed(byQF, warn), nil
}

// ParseSourcesConcurrent parses sources into quarter -> family -> Table using errgroup with
// SetLimit for bounded parallelism, cancelling on first error. Empty/unparseable sources
// are dropped (after recording a warning), matching the Python by_quarter_family build.
func ParseSourcesConcurrent(ctx context.Context, srcs []Source, limit int, warn *Warnings) (map[string]map[string]*Table, error) {
	var (
		mu  sync.Mutex
		out = map[string]map[string]*Table{}
	)
	g, gctx := errgroup.WithContext(ctx)
	if limit > 0 {
		g.SetLimit(limit)
	}
	for _, s := range srcs {
		s := s
		g.Go(func() error {
			if err := gctx.Err(); err != nil {
				return err
			}
			tbl := ParseSource(s, warn)
			if tbl == nil || len(tbl.Rows) == 0 {
				return nil
			}
			mu.Lock()
			if out[s.Quarter] == nil {
				out[s.Quarter] = map[string]*Table{}
			}
			out[s.Quarter][s.Family] = tbl
			mu.Unlock()
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}
	return out, nil
}

// ExtractParsed runs the join/reduce pipeline over already-parsed tables (mirrors the
// quarter loop + _reduce_cases + _select_delete_audit in FAERSASCIIExtractor.extract).
func ExtractParsed(byQF map[string]map[string]*Table, warn *Warnings) Result {
	quarters := make([]string, 0, len(byQF))
	for q := range byQF {
		quarters = append(quarters, q)
	}
	// Most-recent-first (mirrors sorted(by_quarter_family, reverse=True)).
	sort.Slice(quarters, func(i, j int) bool { return quarters[i] > quarters[j] })

	var perQuarter [][]Case
	var deleteTables []*Table
	for _, q := range quarters {
		families := byQF[q]
		if del := families["DELETE"]; del != nil && len(del.Rows) > 0 {
			deleteTables = append(deleteTables, del)
		}
		deleted := DeletedPrimaryIDs(families["DELETE"])
		perQuarter = append(perQuarter, BuildQuarterCases(families, q, deleted, warn))
	}
	kept, dedupAudit := ReduceCases(perQuarter)
	return Result{
		Cases:       kept,
		DeleteAudit: buildDeleteAudit(deleteTables),
		DedupAudit:  dedupAudit,
		Normalized:  byQF,
		Quarters:    quarters,
	}
}

// buildDeleteAudit projects the DELETE tables to the delete-audit columns, sorted by
// auditSortKey (mirrors faers_ascii._select_delete_audit).
func buildDeleteAudit(deleteTables []*Table) []DeleteAuditRow {
	var rows []DeleteAuditRow
	for _, t := range deleteTables {
		q := indexOf(t.Columns, "quarter")
		pid := indexOf(t.Columns, "primaryid")
		cid := indexOf(t.Columns, "caseid")
		sf := indexOf(t.Columns, "source_file")
		srid := indexOf(t.Columns, "source_record_id")
		for _, r := range t.Rows {
			rows = append(rows, DeleteAuditRow{cell(r, q), cell(r, pid), cell(r, cid), cell(r, sf), cell(r, srid)})
		}
	}
	sort.SliceStable(rows, func(i, j int) bool {
		a, b := rows[i], rows[j]
		if a.Quarter != b.Quarter {
			return a.Quarter < b.Quarter
		}
		return a.PrimaryID < b.PrimaryID
	})
	return rows
}

// WriteCasesTSV writes the uncompressed faers_cases.tsv contract (header + rows,
// tab-separated, LF line endings, "" for missing) to w. It mirrors schemas.write_tsv on the
// FAERS_CASES_COLUMNS projection.
func WriteCasesTSV(w io.Writer, cases []Case) error {
	bw := bufio.NewWriter(w)
	if _, err := bw.WriteString(strings.Join(CasesTSVColumns, "\t") + "\n"); err != nil {
		return err
	}
	for _, c := range cases {
		if _, err := bw.WriteString(strings.Join(c.tsvRow(), "\t") + "\n"); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// WriteDeleteAuditTSV writes the delete-audit table as uncompressed TSV.
func WriteDeleteAuditTSV(w io.Writer, rows []DeleteAuditRow) error {
	bw := bufio.NewWriter(w)
	if _, err := bw.WriteString(strings.Join(DeleteAuditColumns, "\t") + "\n"); err != nil {
		return err
	}
	for _, r := range rows {
		line := strings.Join([]string{r.Quarter, r.PrimaryID, r.CaseID, r.SourceFile, r.SourceRecordID}, "\t")
		if _, err := bw.WriteString(line + "\n"); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// WriteDedupAuditTSV writes the cross-quarter dedup-audit table as uncompressed TSV.
func WriteDedupAuditTSV(w io.Writer, rows []DedupAuditRow) error {
	bw := bufio.NewWriter(w)
	if _, err := bw.WriteString(strings.Join(DedupAuditColumns, "\t") + "\n"); err != nil {
		return err
	}
	for _, r := range rows {
		line := strings.Join([]string{r.Quarter, r.PrimaryID, r.CaseID, r.DedupKey, r.WinningQuarter, r.SourceFile}, "\t")
		if _, err := bw.WriteString(line + "\n"); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// tsvRow projects a Case to the public TSV column order (CasesTSVColumns).
func (c Case) tsvRow() []string {
	return []string{
		c.Quarter, c.PrimaryID, c.CaseID, c.Source, c.OccpCod, c.ReporterCountry,
		c.Drugname, c.Ingredient, c.Nda, c.Indication, c.Effects,
	}
}

// dedupSubsetKey is the intra-quarter exact-row dedup key (mirrors faers_ascii._DEDUP_SUBSET).
func (c Case) dedupSubsetKey() string {
	return strings.Join([]string{
		c.Quarter, c.PrimaryID, c.CaseID, c.Source, c.OccpCod, c.ReporterCountry,
		c.Drugname, c.Ingredient, c.Nda, c.Indication, c.Effects,
	}, "\x1f")
}

// dedupKey is the cross-quarter dedup key: caseid when present, else primaryid.
func (c Case) dedupKey() string {
	if c.CaseID != "" {
		return c.CaseID
	}
	return c.PrimaryID
}

// normalizeNDA strips non-digits then leading zeroes (mirrors
// nda_num.str.replace_all(r"\D", "").str.strip_chars_start("0")); it joins Drugs@FDA ApplNo.
func normalizeNDA(raw string) string {
	var b strings.Builder
	b.Grow(len(raw))
	for _, r := range raw {
		if r >= '0' && r <= '9' {
			b.WriteRune(r)
		}
	}
	return strings.TrimLeft(b.String(), "0")
}

// sortCases sorts case rows deterministically by caseSortKey (stable).
func sortCases(cases []Case) {
	sort.SliceStable(cases, func(i, j int) bool {
		a, b := cases[i], cases[j]
		if a.PrimaryID != b.PrimaryID {
			return a.PrimaryID < b.PrimaryID
		}
		if a.DrugSeq != b.DrugSeq {
			return a.DrugSeq < b.DrugSeq
		}
		return a.Indication < b.Indication
	})
}

// sortAudit sorts dedup-audit rows deterministically by auditSortKey (stable).
func sortAudit(rows []DedupAuditRow) {
	sort.SliceStable(rows, func(i, j int) bool {
		a, b := rows[i], rows[j]
		if a.Quarter != b.Quarter {
			return a.Quarter < b.Quarter
		}
		return a.PrimaryID < b.PrimaryID
	})
}

// indexOf returns the index of name in cols, or -1.
func indexOf(cols []string, name string) int {
	for i, c := range cols {
		if c == name {
			return i
		}
	}
	return -1
}

// containsString reports whether cols contains name.
func containsString(cols []string, name string) bool {
	return indexOf(cols, name) >= 0
}

// cell returns row[idx], or "" when idx is out of range (missing column).
func cell(row []string, idx int) string {
	if idx < 0 || idx >= len(row) {
		return ""
	}
	return row[idx]
}
