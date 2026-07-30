// Package drugsfda parses the FDA Drugs@FDA tab-delimited data files (the media/89850
// ZIP contents — Products.txt / Applications.txt / Submissions.txt — or their fixture
// mirrors) into normalized products/applications/submissions tables plus name/ingredient/
// NDC/marketing-status lookup tables, and writes the uncompressed TSV source-section
// tables for Tablassert handoff.
//
// It is the Go mirror of src/dakp_pipeline/extract/drugsfda_products.py (Milestone 3) and
// is byte-for-byte compatible with it: the same inputs produce the same TSV bytes whether
// extracted by Python (polars) or Go (see testdata/golden, computed with the Python
// reference). Two Python behaviors are replicated exactly because they are visible in the
// output bytes:
//
//   - source_record_id is the Drugs@FDA STRING form (drugsfda:product:NDA12345:001,
//     drugsfda:application:NDA12345, drugsfda:submission:NDA12345:1) — NOT a b3:<hex>
//     hash. The Python reference asserts these start with "drugsfda:product:" etc., so
//     parity requires the string form (pipeline.SourceRecordID's b3 derivation is used by
//     other sources, not Drugs@FDA).
//   - The TSV writer mirrors polars write_csv(separator="\t"): a field is quoted iff it is
//     empty or contains a quote, tab, CR, or LF; embedded quotes are doubled. In
//     particular an EMPTY field is written as the two literal characters "" (not nothing).
//
// Application-number normalization ports the legacy FAERS/bin/drug2indi.pl readNDAproducts
// semantics (s/^(NDA|BLA|ANDA)0*(.+)/): the raw APPLICATIONNUMBER is preserved AND both
// normalized forms are kept — digits with leading zeroes kept (appl_no, e.g. 012345) and
// leading zeroes stripped (appl_no_stripped, e.g. 12345) — so NDA/BLA/ANDA variants join
// robustly with FAERS nda values regardless of padding.
package drugsfda

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// --- normalized column contracts (mirror drugsfda_products.py) -------------------

// ProductsColumns is the ordered column layout of the products source-section TSV.
var ProductsColumns = []string{
	"source_record_id",
	"source_file",
	"appl_no_raw",
	"appl_type",
	"appl_no",
	"appl_no_stripped",
	"product_no",
	"drug_name",
	"active_ingredient",
	"form",
	"route",
	"strength",
	"reference_drug",
	"reference_standard",
	"product_ndc",
	"marketing_status_name",
}

// ApplicationsColumns is the ordered column layout of the applications table.
var ApplicationsColumns = []string{
	"source_record_id",
	"source_file",
	"appl_no_raw",
	"appl_type",
	"appl_no",
	"appl_no_stripped",
	"sponsor_name",
	"common_or_original_name",
	"submission_classification",
	"orphan_status",
}

// SubmissionsColumns is the ordered column layout of the submissions table.
var SubmissionsColumns = []string{
	"source_record_id",
	"source_file",
	"appl_no_raw",
	"appl_type",
	"appl_no",
	"appl_no_stripped",
	"submission_type",
	"submission_no",
	"submission_status",
	"submission_status_date",
	"submission_notes",
}

// LookupsColumns is the ordered column layout of the lookup table.
var LookupsColumns = []string{"lookup_type", "term", "appl_no", "appl_no_stripped", "appl_type"}

// --- source-column alias matching ------------------------------------------------

// fieldAlias maps one canonical field to its accepted source-column spellings, in
// priority order. Matching is case/space/underscore-insensitive (see normKey). Order
// matters: aliases are tried in slice order and canonical fields are claimed in slice
// order, mirroring the Python dict iteration that drives _rename_to_canonical.
type fieldAlias struct {
	canonical string
	aliases   []string
}

var productFieldAliases = []fieldAlias{
	{"appl_no_raw", []string{"applicationnumber"}},
	{"appl_type", []string{"appltype", "applicationtype"}},
	{"appl_no", []string{"applno"}},
	{"product_no", []string{"productno"}},
	{"drug_name", []string{"drugname", "proprietaryname"}},
	{"active_ingredient", []string{"activeingredient", "nonproprietaryname"}},
	{"form", []string{"form", "dosageformname"}},
	{"route", []string{"route", "routename"}},
	{"strength", []string{"strength"}},
	{"reference_drug", []string{"referencedrug"}},
	{"reference_standard", []string{"referencestandard"}},
	{"product_ndc", []string{"productndc"}},
	{"marketing_status_name", []string{"marketingstatusname", "marketingstatusdescription"}},
}

var applicationFieldAliases = []fieldAlias{
	{"appl_no_raw", []string{"applicationnumber"}},
	{"appl_type", []string{"appltype"}},
	{"appl_no", []string{"applno"}},
	{"sponsor_name", []string{"sponsorname", "labelername"}},
	{"common_or_original_name", []string{"commonororiginalname"}},
	{"submission_classification", []string{"submissionclassification"}},
	{"orphan_status", []string{"orphanstatus"}},
}

var submissionFieldAliases = []fieldAlias{
	{"appl_no_raw", []string{"applicationnumber"}},
	{"appl_type", []string{"appltype"}},
	{"appl_no", []string{"applno"}},
	{"submission_type", []string{"submissiontype"}},
	{"submission_no", []string{"submissionno"}},
	{"submission_status", []string{"submissionstatus"}},
	{"submission_status_date", []string{"submissionstatusdate"}},
	{"submission_notes", []string{"submissionspublicnotes"}},
}

// ingredientSeparator splits multi-ingredient Drugs@FDA ActiveIngredient values
// (e.g. "EZETIMIBE; SIMVASTATIN").
const ingredientSeparator = ";"

// --- shared types ----------------------------------------------------------------

// Row is one normalized output record keyed by canonical column name. WriteTSV emits
// cells in the relevant *Columns order; absent keys write as empty.
type Row map[string]string

// Table is one parsed Drugs@FDA source table: its originating filename (used as the
// source_file column) plus the raw header and data rows (string cells, untrimmed).
type Table struct {
	SourceName string
	Header     []string
	Rows       [][]string
}

// Warning is one deterministic parse-warning record (mirrors the Python warnings dicts).
type Warning struct {
	Table      string `json:"table"`
	SourceFile string `json:"source_file,omitempty"`
	Message    string `json:"message"`
}

// --- column normalization --------------------------------------------------------

// normKey is the case/space/underscore-insensitive key used to match source columns to
// aliases (mirrors _norm_key): trim, lowercase, strip a leading BOM, drop spaces and
// underscores.
func normKey(column string) string {
	s := strings.TrimSpace(column)
	s = strings.ToLower(s)
	s = strings.TrimLeft(s, "\ufeff")
	s = strings.ReplaceAll(s, " ", "")
	s = strings.ReplaceAll(s, "_", "")
	return s
}

// buildFieldIndex maps canonical field name -> source-column index for one header,
// faithfully replicating Python _rename_to_canonical: the first source column matching
// each alias (in priority order) claims the canonical field, a source column already
// claimed is not reused, and (mirroring the Python guard) a canonical name that is itself
// an already-processed source column is skipped.
func buildFieldIndex(header []string, aliases []fieldAlias) map[string]int {
	// normKey -> first source-column index (mirrors src_to_orig.setdefault).
	srcToIdx := make(map[string]int, len(header))
	for i, col := range header {
		k := normKey(col)
		if _, ok := srcToIdx[k]; !ok {
			srcToIdx[k] = i
		}
	}
	renameKeys := make(map[string]bool)   // source-column names used as python rename keys
	renameValues := make(map[string]bool) // canonical names used as python rename values
	fieldToIdx := make(map[string]int, len(aliases))
	for _, fa := range aliases {
		if renameKeys[fa.canonical] { // if canonical in rename (keys)
			continue
		}
		for _, alias := range fa.aliases {
			idx, ok := srcToIdx[normKey(alias)]
			if !ok {
				continue
			}
			origCol := header[idx]
			if renameValues[origCol] { // orig not in rename.values()
				continue
			}
			fieldToIdx[fa.canonical] = idx
			renameKeys[origCol] = true
			renameValues[fa.canonical] = true
			break
		}
	}
	return fieldToIdx
}

// fieldGetter returns a closure reading canonical fields from a raw row by index. Missing
// fields and short rows yield "" (mirrors _field returning "" for absent/null values).
func fieldGetter(fieldToIdx map[string]int, raw []string) func(string) string {
	return func(field string) string {
		idx, ok := fieldToIdx[field]
		if !ok || idx >= len(raw) {
			return ""
		}
		return raw[idx]
	}
}

// --- application-number normalization (ports legacy readNDAproducts) -------------

// applPrefixRe splits a combined APPLICATIONNUMBER into (type, digits-with-zeroes),
// mirroring ^(NDA|BLA|ANDA)\s*(\d+) (IGNORECASE, anchored at start).
var applPrefixRe = regexp.MustCompile(`(?i)^(NDA|BLA|ANDA)\s*(\d+)`)

// digitsOnly keeps only ASCII decimal digits (mirrors "".join(ch for ch in v if
// ch.isdigit()) for the ASCII digits that occur in Drugs@FDA/FAERS data).
func digitsOnly(value string) string {
	var b strings.Builder
	for i := 0; i < len(value); i++ {
		if c := value[i]; c >= '0' && c <= '9' {
			b.WriteByte(c)
		}
	}
	return b.String()
}

// parseCombined splits a combined APPLICATIONNUMBER into (appl_type, digits_with_zeroes).
// A recognized NDA/BLA/ANDA prefix yields (upper(prefix), digits); otherwise ("",
// digits_only(value)).
func parseCombined(value string) (prefix, digits string) {
	text := strings.TrimSpace(value)
	if text == "" {
		return "", ""
	}
	if m := applPrefixRe.FindStringSubmatch(text); m != nil {
		return strings.ToUpper(m[1]), m[2]
	}
	return "", digitsOnly(text)
}

// applFields is the normalized application-number quadruple.
type applFields struct {
	Raw      string // appl_no_raw: {type}{digits-with-zeroes} (or just digits if no type)
	Type     string // appl_type: NDA | BLA | ANDA | "" (uppercased)
	No       string // appl_no: digits with leading zeroes preserved
	Stripped string // appl_no_stripped: leading zeroes removed (all-zero/empty kept as-is)
}

// normalizeApplFields returns the (raw, type, no, stripped) quadruple, handling both
// NDC-style combined APPLICATIONNUMBER ("NDA012345") and Drugs@FDA split ApplType+ApplNo
// ("NDA", "012345"). appl_no keeps leading zeroes; appl_no_stripped removes them —
// mirroring legacy s/^(NDA|BLA|ANDA)0*(.+)/.
func normalizeApplFields(applNoRawSrc, applTypeSrc, applNoSrc string) applFields {
	rawSrc := strings.TrimSpace(applNoRawSrc)
	atype := strings.ToUpper(strings.TrimSpace(applTypeSrc))
	ano := strings.TrimSpace(applNoSrc)

	var digits string
	if rawSrc != "" {
		prefix, d := parseCombined(rawSrc)
		if prefix != "" && atype == "" {
			atype = prefix
		}
		digits = d
		if digits == "" {
			digits = digitsOnly(ano)
		}
	} else {
		digits = digitsOnly(ano)
	}

	applNo := digits
	stripped := strings.TrimLeft(applNo, "0")
	applNoStripped := stripped
	if applNoStripped == "" {
		applNoStripped = applNo // keep all-zero/empty as-is
	}
	applNoRaw := applNo
	if atype != "" {
		applNoRaw = atype + applNo
	}
	return applFields{Raw: applNoRaw, Type: atype, No: applNo, Stripped: applNoStripped}
}

// --- record ids (mirror _product_record_id / _record_id) -------------------------

func productRecordID(applType, applNoStripped, productNo, productNDC string, rowIndex int) string {
	if applNoStripped != "" {
		pn := productNo
		if pn == "" {
			pn = "NA"
		}
		return fmt.Sprintf("drugsfda:product:%s%s:%s", applType, applNoStripped, pn)
	}
	if productNDC != "" {
		return "drugsfda:product:ndc:" + productNDC
	}
	return fmt.Sprintf("drugsfda:product:row:%d", rowIndex)
}

func recordID(kind, applType, applNoStripped string, rowIndex int, suffix string) string {
	if applNoStripped != "" {
		base := fmt.Sprintf("drugsfda:%s:%s%s", kind, applType, applNoStripped)
		if suffix != "" {
			return base + ":" + suffix
		}
		return base
	}
	return fmt.Sprintf("drugsfda:%s:row:%d", kind, rowIndex)
}

// --- per-table builders ----------------------------------------------------------

// BuildProducts normalizes a products Table into Rows (ProductsColumns) plus row-level
// warnings. Row order is preserved (source order); rowIndex is the 1-based source line
// (header is line 1, first data row is line 2), matching the Python enumerate(start=2).
func BuildProducts(tbl Table) ([]Row, []Warning) {
	fieldToIdx := buildFieldIndex(tbl.Header, productFieldAliases)
	rows := make([]Row, 0, len(tbl.Rows))
	var warnings []Warning
	for i, raw := range tbl.Rows {
		rowIndex := i + 2
		get := fieldGetter(fieldToIdx, raw)
		af := normalizeApplFields(get("appl_no_raw"), get("appl_type"), get("appl_no"))
		if af.No == "" {
			warnings = append(warnings, Warning{Table: "products", SourceFile: tbl.SourceName, Message: fmt.Sprintf("row %d: missing application number", rowIndex)})
		}
		productNo := strings.TrimSpace(get("product_no"))
		productNDC := strings.TrimSpace(get("product_ndc"))
		rows = append(rows, Row{
			"source_record_id":      productRecordID(af.Type, af.Stripped, productNo, productNDC, rowIndex),
			"source_file":           tbl.SourceName,
			"appl_no_raw":           af.Raw,
			"appl_type":             af.Type,
			"appl_no":               af.No,
			"appl_no_stripped":      af.Stripped,
			"product_no":            productNo,
			"drug_name":             strings.TrimSpace(get("drug_name")),
			"active_ingredient":     strings.TrimSpace(get("active_ingredient")),
			"form":                  strings.TrimSpace(get("form")),
			"route":                 strings.TrimSpace(get("route")),
			"strength":              strings.TrimSpace(get("strength")),
			"reference_drug":        strings.TrimSpace(get("reference_drug")),
			"reference_standard":    strings.TrimSpace(get("reference_standard")),
			"product_ndc":           productNDC,
			"marketing_status_name": strings.TrimSpace(get("marketing_status_name")),
		})
	}
	return rows, warnings
}

// BuildApplications normalizes an applications Table into Rows (ApplicationsColumns).
func BuildApplications(tbl Table) ([]Row, []Warning) {
	fieldToIdx := buildFieldIndex(tbl.Header, applicationFieldAliases)
	rows := make([]Row, 0, len(tbl.Rows))
	var warnings []Warning
	for i, raw := range tbl.Rows {
		rowIndex := i + 2
		get := fieldGetter(fieldToIdx, raw)
		af := normalizeApplFields(get("appl_no_raw"), get("appl_type"), get("appl_no"))
		if af.No == "" {
			warnings = append(warnings, Warning{Table: "applications", SourceFile: tbl.SourceName, Message: fmt.Sprintf("row %d: missing application number", rowIndex)})
		}
		rows = append(rows, Row{
			"source_record_id":          recordID("application", af.Type, af.Stripped, rowIndex, ""),
			"source_file":               tbl.SourceName,
			"appl_no_raw":               af.Raw,
			"appl_type":                 af.Type,
			"appl_no":                   af.No,
			"appl_no_stripped":          af.Stripped,
			"sponsor_name":              strings.TrimSpace(get("sponsor_name")),
			"common_or_original_name":   strings.TrimSpace(get("common_or_original_name")),
			"submission_classification": strings.TrimSpace(get("submission_classification")),
			"orphan_status":             strings.TrimSpace(get("orphan_status")),
		})
	}
	return rows, warnings
}

// BuildSubmissions normalizes a submissions Table into Rows (SubmissionsColumns). The real
// Submissions.txt carries no ApplType, so appl_type is inherited from products/applications
// via applTypeMap (keyed by appl_no_stripped); the inherited type also rebuilds appl_no_raw.
func BuildSubmissions(tbl Table, applTypeMap map[string]string) ([]Row, []Warning) {
	fieldToIdx := buildFieldIndex(tbl.Header, submissionFieldAliases)
	rows := make([]Row, 0, len(tbl.Rows))
	var warnings []Warning
	for i, raw := range tbl.Rows {
		rowIndex := i + 2
		get := fieldGetter(fieldToIdx, raw)
		af := normalizeApplFields(get("appl_no_raw"), get("appl_type"), get("appl_no"))
		if af.No == "" {
			warnings = append(warnings, Warning{Table: "submissions", SourceFile: tbl.SourceName, Message: fmt.Sprintf("row %d: missing application number", rowIndex)})
		}
		applType := af.Type
		applNoRaw := af.Raw
		if applType == "" && af.Stripped != "" {
			if inherited := applTypeMap[af.Stripped]; inherited != "" {
				applType = inherited
				if af.No != "" {
					applNoRaw = inherited + af.No
				}
			}
		}
		submissionNo := strings.TrimSpace(get("submission_no"))
		rows = append(rows, Row{
			"source_record_id":       recordID("submission", applType, af.Stripped, rowIndex, submissionNo),
			"source_file":            tbl.SourceName,
			"appl_no_raw":            applNoRaw,
			"appl_type":              applType,
			"appl_no":                af.No,
			"appl_no_stripped":       af.Stripped,
			"submission_type":        strings.TrimSpace(get("submission_type")),
			"submission_no":          submissionNo,
			"submission_status":      strings.TrimSpace(get("submission_status")),
			"submission_status_date": strings.TrimSpace(get("submission_status_date")),
			"submission_notes":       strings.TrimSpace(get("submission_notes")),
		})
	}
	return rows, warnings
}

// BuildLookups builds deduplicated name/ingredient/NDC/marketing-status -> appl_no lookup
// Rows (LookupsColumns) from already-normalized products rows. Per product row (in order)
// it emits proprietary_name, nonproprietary_name, each split ingredient, product_ndc, and
// marketing_status candidates; duplicates are dropped by (lookup_type, casefolded term,
// appl_no_stripped). casefold is approximated by strings.ToLower (identical for the ASCII
// terms in Drugs@FDA data; only exotic Unicode case-folding differs, affecting dedup only).
func BuildLookups(products []Row) []Row {
	type seenKey struct {
		lookupType string
		termFold   string
		stripped   string
	}
	seen := make(map[seenKey]bool)
	var rows []Row
	for _, rec := range products {
		applNo := rec["appl_no"]
		stripped := rec["appl_no_stripped"]
		applType := rec["appl_type"]
		if stripped == "" {
			continue
		}
		var candidates [][2]string
		if drugName := rec["drug_name"]; drugName != "" {
			candidates = append(candidates, [2]string{"proprietary_name", drugName})
		}
		if activeIngredient := rec["active_ingredient"]; activeIngredient != "" {
			candidates = append(candidates, [2]string{"nonproprietary_name", activeIngredient})
			for _, part := range strings.Split(activeIngredient, ingredientSeparator) {
				if ingredient := strings.TrimSpace(part); ingredient != "" {
					candidates = append(candidates, [2]string{"ingredient", ingredient})
				}
			}
		}
		if productNDC := rec["product_ndc"]; productNDC != "" {
			candidates = append(candidates, [2]string{"product_ndc", productNDC})
		}
		if marketingStatus := rec["marketing_status_name"]; marketingStatus != "" {
			candidates = append(candidates, [2]string{"marketing_status", marketingStatus})
		}
		for _, c := range candidates {
			lookupType, term := c[0], c[1]
			key := seenKey{lookupType, strings.ToLower(term), stripped}
			if seen[key] {
				continue
			}
			seen[key] = true
			rows = append(rows, Row{
				"lookup_type":      lookupType,
				"term":             term,
				"appl_no":          applNo,
				"appl_no_stripped": stripped,
				"appl_type":        applType,
			})
		}
	}
	return rows
}

// FillApplTypeMap records appl_no_stripped -> appl_type from normalized rows (first writer
// wins). Call with products first, then applications, so submissions inherit appl_type with
// products taking precedence (mirrors _fill_appl_type_map).
func FillApplTypeMap(m map[string]string, rows []Row) {
	for _, rec := range rows {
		stripped := rec["appl_no_stripped"]
		applType := rec["appl_type"]
		if stripped != "" && applType != "" {
			if _, ok := m[stripped]; !ok {
				m[stripped] = applType
			}
		}
	}
}

// --- orchestration ---------------------------------------------------------------

// Tables holds the recognized input tables; any may be nil if absent from the inputs.
type Tables struct {
	Products     *Table
	Applications *Table
	Submissions  *Table
}

// Result is the full normalized extraction output.
type Result struct {
	Products         []Row
	Applications     []Row
	Submissions      []Row
	Lookups          []Row
	Warnings         []Warning
	HaveProducts     bool
	HaveApplications bool
	HaveSubmissions  bool
}

// Extract normalizes the recognized tables into products/applications/submissions/lookups
// rows plus deterministic warnings, mirroring DrugsFDAProductsExtractor.extract (minus the
// parquet/manifest registration, which is Python-orchestrator concern). The appl_type map
// is filled from products then applications so submissions inherit appl_type.
func Extract(tables Tables) Result {
	var res Result
	applTypeMap := make(map[string]string)

	if tables.Products != nil {
		res.HaveProducts = true
		res.Products, res.Warnings = BuildProducts(*tables.Products)
		FillApplTypeMap(applTypeMap, res.Products)
	} else {
		res.Warnings = append(res.Warnings, Warning{Table: "products", Message: "no Products table found in inputs"})
	}

	if tables.Applications != nil {
		res.HaveApplications = true
		appRows, appWarnings := BuildApplications(*tables.Applications)
		res.Applications = appRows
		res.Warnings = append(res.Warnings, appWarnings...)
		FillApplTypeMap(applTypeMap, res.Applications)
	} else {
		res.Warnings = append(res.Warnings, Warning{Table: "applications", Message: "no Applications table found in inputs"})
	}

	if tables.Submissions != nil {
		res.HaveSubmissions = true
		subRows, subWarnings := BuildSubmissions(*tables.Submissions, applTypeMap)
		res.Submissions = subRows
		res.Warnings = append(res.Warnings, subWarnings...)
	} else {
		res.Warnings = append(res.Warnings, Warning{Table: "submissions", Message: "no Submissions table found in inputs"})
	}

	if res.HaveProducts {
		res.Lookups = BuildLookups(res.Products)
	}
	return res
}

// --- input parsing ---------------------------------------------------------------

// Classify maps a Drugs@FDA table filename to its table key ("products", "applications",
// "submissions") or "" if unrecognized. Matches by stem suffix so it accepts both the real
// files (Products.txt) and fixture mirrors (drugsfda_products.tsv) while rejecting
// sub-tables like SubmissionPropertyType.txt (mirrors _table_key).
func Classify(filename string) string {
	stem := strings.ToLower(strings.TrimSuffix(filepath.Base(filename), filepath.Ext(filename)))
	switch {
	case strings.HasSuffix(stem, "products") || stem == "product":
		return "products"
	case strings.HasSuffix(stem, "applications") || stem == "application":
		return "applications"
	case strings.HasSuffix(stem, "submissions") || stem == "submission":
		return "submissions"
	default:
		return ""
	}
}

// ParseTSV streams a tab-delimited file into a Table (all cells UTF-8 strings, untrimmed),
// mirroring polars read_csv(separator="\t", infer_schema_length=0): every column is a
// string so leading zeroes and NDA/BLA/ANDA prefixes survive. SourceName is set to the
// file's base name (the source_file column). Ragged rows are tolerated (short rows read as
// empty fields); blank trailing lines are ignored.
func ParseTSV(path string) (Table, error) {
	f, err := os.Open(path)
	if err != nil {
		return Table{}, err
	}
	defer f.Close()
	return ParseTSVReader(bufio.NewReader(f), filepath.Base(path))
}

// ParseTSVReader parses tab-delimited TSV from r into a Table with the given source name.
func ParseTSVReader(r io.Reader, sourceName string) (Table, error) {
	cr := csv.NewReader(r)
	cr.Comma = '\t'
	cr.FieldsPerRecord = -1 // tolerate ragged rows; fieldGetter bounds-checks per row
	cr.LazyQuotes = true    // tolerate stray quotes (polars is lenient too)

	var header []string
	var rows [][]string
	for {
		rec, err := cr.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return Table{}, fmt.Errorf("parse %s: %w", sourceName, err)
		}
		if len(rec) == 0 || (len(rec) == 1 && rec[0] == "") {
			continue // skip blank lines
		}
		if header == nil {
			header = rec
			continue
		}
		rows = append(rows, rec)
	}
	if header == nil {
		return Table{}, fmt.Errorf("parse %s: empty TSV (no header row)", sourceName)
	}
	return Table{SourceName: sourceName, Header: header, Rows: rows}, nil
}

// --- TSV writing (polars write_csv byte-compatible) ------------------------------

// encodeTSVCell renders one cell exactly as polars write_csv(separator="\t") does: quote
// iff the value is empty or contains a quote, tab, CR, or LF; embedded quotes are doubled.
// Notably an empty value becomes the two literal characters "" (verified against polars).
func encodeTSVCell(s string) string {
	if s != "" && !strings.ContainsAny(s, "\"\t\n\r") {
		return s
	}
	return `"` + strings.ReplaceAll(s, `"`, `""`) + `"`
}

// WriteTSV writes rows as an uncompressed TSV (header + data rows, "\n" line endings,
// trailing newline) in the given column order, byte-compatible with polars
// write_csv(separator="\t"). Output is buffered for streaming writes.
func WriteTSV(w io.Writer, columns []string, rows []Row) error {
	bw := bufio.NewWriter(w)
	if err := writeTSVRow(bw, columns); err != nil {
		return err
	}
	cells := make([]string, len(columns))
	for _, row := range rows {
		for i, col := range columns {
			cells[i] = encodeTSVCell(row[col])
		}
		if _, err := bw.WriteString(strings.Join(cells, "\t")); err != nil {
			return err
		}
		if err := bw.WriteByte('\n'); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// writeTSVRow writes the header row (column names are never quoted: they contain no
// special characters).
func writeTSVRow(bw *bufio.Writer, columns []string) error {
	if _, err := bw.WriteString(strings.Join(columns, "\t")); err != nil {
		return err
	}
	return bw.WriteByte('\n')
}

// WriteTSVFile writes rows to path (creating parent directories) as an uncompressed TSV in
// the given column order. Returns the number of data rows written.
func WriteTSVFile(path string, columns []string, rows []Row) (int, error) {
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return 0, err
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return 0, err
	}
	if err := WriteTSV(f, columns, rows); err != nil {
		f.Close()
		return 0, err
	}
	if err := f.Close(); err != nil {
		return 0, err
	}
	return len(rows), nil
}
