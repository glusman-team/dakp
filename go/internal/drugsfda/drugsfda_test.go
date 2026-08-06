package drugsfda

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/glusman-team/dakp/go/internal/blake3store"
)

// The testdata/golden/*.tsv fixtures were generated with the PYTHON reference
// (src/dakp_pipeline/extract/drugsfda_products.py + schemas.write_tsv, polars 1.43.1) from
// the same input fixtures (byte-identical copies of tests/fixtures/pipeline/drugsfda/*.tsv).
// TestParityGoldenTSV asserts the Go output matches them byte-for-byte — the same
// cross-language parity pattern blake3store uses for its golden fixtures.

// --- helpers ---------------------------------------------------------------------

func loadTable(t *testing.T, name string) Table {
	t.Helper()
	tbl, err := ParseTSV(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("ParseTSV(%s): %v", name, err)
	}
	return tbl
}

func extractFixtures(t *testing.T) Result {
	t.Helper()
	products := loadTable(t, "drugsfda_products.tsv")
	applications := loadTable(t, "drugsfda_applications.tsv")
	submissions := loadTable(t, "drugsfda_submissions.tsv")
	return Extract(Tables{Products: &products, Applications: &applications, Submissions: &submissions})
}

func renderTSV(t *testing.T, columns []string, rows []Row) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := WriteTSV(&buf, columns, rows); err != nil {
		t.Fatalf("WriteTSV: %v", err)
	}
	return buf.Bytes()
}

func readGolden(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "golden", name))
	if err != nil {
		t.Fatalf("read golden %s: %v", name, err)
	}
	return b
}

func assertTermSet(t *testing.T, got map[string]bool, want ...string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("term set = %v, want exactly %v", got, want)
	}
	for _, w := range want {
		if !got[w] {
			t.Errorf("term set missing %q (got %v)", w, got)
		}
	}
}

// --- column contracts (guard against drift from the Python column layout) --------

func TestColumnContractsMatchPython(t *testing.T) {
	wantProducts := []string{
		"source_record_id", "source_file", "appl_no_raw", "appl_type", "appl_no",
		"appl_no_stripped", "product_no", "drug_name", "active_ingredient", "form",
		"route", "strength", "reference_drug", "reference_standard", "product_ndc",
		"marketing_status_name",
	}
	wantApplications := []string{
		"source_record_id", "source_file", "appl_no_raw", "appl_type", "appl_no",
		"appl_no_stripped", "sponsor_name", "common_or_original_name",
		"submission_classification", "orphan_status",
	}
	wantSubmissions := []string{
		"source_record_id", "source_file", "appl_no_raw", "appl_type", "appl_no",
		"appl_no_stripped", "submission_type", "submission_no", "submission_status",
		"submission_status_date", "submission_notes",
	}
	wantLookups := []string{"lookup_type", "term", "appl_no", "appl_no_stripped", "appl_type"}

	for _, c := range []struct {
		name      string
		got, want []string
	}{
		{"products", ProductsColumns, wantProducts},
		{"applications", ApplicationsColumns, wantApplications},
		{"submissions", SubmissionsColumns, wantSubmissions},
		{"lookups", LookupsColumns, wantLookups},
	} {
		if strings.Join(c.got, "\t") != strings.Join(c.want, "\t") {
			t.Errorf("%s columns = %v, want %v", c.name, c.got, c.want)
		}
	}
}

// --- application-number normalization (raw + both normalized forms) --------------

func TestNormalizeApplFields(t *testing.T) {
	cases := []struct {
		name                   string
		rawSrc, typeSrc, noSrc string
		want                   applFields
	}{
		{"NDA leading zero (split)", "", "NDA", "012345", applFields{"NDA012345", "NDA", "012345", "12345"}},
		{"NDA no leading zero (split)", "", "NDA", "207500", applFields{"NDA207500", "NDA", "207500", "207500"}},
		{"BLA (split)", "", "BLA", "125557", applFields{"BLA125557", "BLA", "125557", "125557"}},
		{"ANDA leading zero (split)", "", "ANDA", "075123", applFields{"ANDA075123", "ANDA", "075123", "75123"}},
		{"combined NDA012345", "NDA012345", "", "", applFields{"NDA012345", "NDA", "012345", "12345"}},
		{"combined BLA125557", "BLA125557", "", "", applFields{"BLA125557", "BLA", "125557", "125557"}},
		{"combined lowercase", "nda012345", "", "", applFields{"NDA012345", "NDA", "012345", "12345"}},
		{"combined with space", "NDA 012345", "", "", applFields{"NDA012345", "NDA", "012345", "12345"}},
		{"combined raw beats fallback no", "ANDA075123", "", "999", applFields{"ANDA075123", "ANDA", "075123", "75123"}},
		{"all-zero keeps form", "", "NDA", "000000", applFields{"NDA000000", "NDA", "000000", "000000"}},
		{"digits only no type", "", "", "012345", applFields{"012345", "", "012345", "12345"}},
		{"empty", "", "", "", applFields{"", "", "", ""}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := normalizeApplFields(c.rawSrc, c.typeSrc, c.noSrc); got != c.want {
				t.Errorf("normalizeApplFields(%q,%q,%q) = %+v, want %+v", c.rawSrc, c.typeSrc, c.noSrc, got, c.want)
			}
		})
	}
}

func TestBuildProductsKeepsRawAndBothForms(t *testing.T) {
	res := extractFixtures(t)
	byRaw := make(map[string]Row, len(res.Products))
	for _, r := range res.Products {
		byRaw[r["appl_no_raw"]] = r
	}
	for _, c := range []struct{ raw, typ, no, stripped string }{
		{"NDA012345", "NDA", "012345", "12345"},
		{"NDA207500", "NDA", "207500", "207500"},
		{"BLA125557", "BLA", "125557", "125557"},
		{"ANDA075123", "ANDA", "075123", "75123"},
	} {
		r, ok := byRaw[c.raw]
		if !ok {
			t.Errorf("no product row with appl_no_raw %q", c.raw)
			continue
		}
		if r["appl_type"] != c.typ || r["appl_no"] != c.no || r["appl_no_stripped"] != c.stripped {
			t.Errorf("row %q = (type %q, no %q, stripped %q), want (%q, %q, %q)",
				c.raw, r["appl_type"], r["appl_no"], r["appl_no_stripped"], c.typ, c.no, c.stripped)
		}
	}
	// The combined raw form is always {type}{no} (legacy readNDAproducts preservation).
	for _, r := range res.Products {
		if r["appl_no_raw"] != r["appl_type"]+r["appl_no"] {
			t.Errorf("appl_no_raw %q != type+no %q", r["appl_no_raw"], r["appl_type"]+r["appl_no"])
		}
	}
}

// --- record ids ------------------------------------------------------------------

func TestRecordIDs(t *testing.T) {
	cases := []struct {
		name string
		got  string
		want string
	}{
		{"product full", productRecordID("NDA", "12345", "001", "", 2), "drugsfda:product:NDA12345:001"},
		{"product empty product_no -> NA", productRecordID("NDA", "12345", "", "", 2), "drugsfda:product:NDA12345:NA"},
		{"product ndc fallback", productRecordID("", "", "", "00000-001", 2), "drugsfda:product:ndc:00000-001"},
		{"product row fallback", productRecordID("", "", "", "", 5), "drugsfda:product:row:5"},
		{"application", recordID("application", "BLA", "125557", 3, ""), "drugsfda:application:BLA125557"},
		{"submission with suffix", recordID("submission", "NDA", "12345", 2, "1"), "drugsfda:submission:NDA12345:1"},
		{"submission row fallback", recordID("submission", "", "", 4, ""), "drugsfda:submission:row:4"},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("%s: got %q, want %q", c.name, c.got, c.want)
		}
	}
}

// --- source_record_id format + b3 artifact ids -----------------------------------

// TestSourceRecordIDFormat asserts the Drugs@FDA source_record_id is the stable STRING
// form emitted by the Python reference (drugsfda:product:NDA12345:001 etc.) — NOT a b3
// hash. The Python test test_source_record_ids_are_stable_and_unique asserts these start
// with "drugsfda:product:", so byte-parity requires the string form. (pipeline.SourceRecordID's
// b3:<hex> derivation is the canonical id for OTHER sources; Drugs@FDA uses string forms.)
func TestSourceRecordIDFormat(t *testing.T) {
	res := extractFixtures(t)
	check := func(table string, rows []Row, prefix string) {
		t.Helper()
		seen := make(map[string]bool, len(rows))
		for _, r := range rows {
			id := r["source_record_id"]
			if !strings.HasPrefix(id, prefix) {
				t.Errorf("%s: source_record_id %q does not start with %q", table, id, prefix)
			}
			if seen[id] {
				t.Errorf("%s: duplicate source_record_id %q", table, id)
			}
			seen[id] = true
		}
	}
	check("products", res.Products, "drugsfda:product:")
	check("applications", res.Applications, "drugsfda:application:")
	check("submissions", res.Submissions, "drugsfda:submission:")

	// Exact known ids (parity with the Python reference output).
	if got := res.Products[0]["source_record_id"]; got != "drugsfda:product:NDA12345:001" {
		t.Errorf("products[0] source_record_id = %q, want drugsfda:product:NDA12345:001", got)
	}
	if got := res.Applications[0]["source_record_id"]; got != "drugsfda:application:NDA12345" {
		t.Errorf("applications[0] source_record_id = %q, want drugsfda:application:NDA12345", got)
	}
	if got := res.Submissions[0]["source_record_id"]; got != "drugsfda:submission:NDA12345:1" {
		t.Errorf("submissions[0] source_record_id = %q, want drugsfda:submission:NDA12345:1", got)
	}
}

// TestArtifactIDsAreB3 asserts the content/artifact ids use the canonical b3:<hex> form
// (32-byte / 64-hex BLAKE3) — the b3 format the DAKP pipeline uses for artifact addressing.
func TestArtifactIDsAreB3(t *testing.T) {
	id := blake3store.HashBytes([]byte("drugsfda:product:NDA12345:001"))
	if !strings.HasPrefix(id, "b3:") {
		t.Fatalf("artifact id %q lacks b3: prefix", id)
	}
	if hex := strings.TrimPrefix(id, "b3:"); len(hex) != 64 {
		t.Errorf("artifact id hex length = %d, want 64 (%q)", len(hex), id)
	}
}

// --- TSV writer: polars write_csv byte-compatible quoting ------------------------

func TestEncodeTSVCell(t *testing.T) {
	cases := []struct{ in, want string }{
		{"", `""`}, // empty -> literal "" (polars behavior)
		{"plain", "plain"},
		{"has space", "has space"},
		{`has"quote`, `"has""quote"`},
		{"has\ttab", "\"has\ttab\""},
		{"has\nnewline", "\"has\nnewline\""},
		{"has\rcr", "\"has\rcr\""},
		{"NDA012345", "NDA012345"},
		{"EZETIMIBE; SIMVASTATIN", "EZETIMIBE; SIMVASTATIN"},
		{"00000-001", "00000-001"},
		{"a,b", "a,b"},
	}
	for _, c := range cases {
		if got := encodeTSVCell(c.in); got != c.want {
			t.Errorf("encodeTSVCell(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestTSVHeaderColumns(t *testing.T) {
	res := extractFixtures(t)
	got := renderTSV(t, ProductsColumns, res.Products)
	header, _, _ := strings.Cut(string(got), "\n")
	if want := strings.Join(ProductsColumns, "\t"); header != want {
		t.Errorf("products TSV header = %q, want %q", header, want)
	}
}

// --- lookups ---------------------------------------------------------------------

func TestBuildLookups(t *testing.T) {
	res := extractFixtures(t)
	terms := func(lookupType, stripped string) map[string]bool {
		m := map[string]bool{}
		for _, r := range res.Lookups {
			if r["lookup_type"] == lookupType && r["appl_no_stripped"] == stripped {
				m[r["term"]] = true
			}
		}
		return m
	}
	assertTermSet(t, terms("proprietary_name", "125557"), "Keytruda")
	assertTermSet(t, terms("nonproprietary_name", "12345"), "EXAMPLESTATIN")
	// Multi-ingredient product: both ingredients map to ANDA 75123.
	assertTermSet(t, terms("ingredient", "75123"), "EZETIMIBE", "SIMVASTATIN")

	// No product_ndc lookup type when the fixture has no NDC column.
	for _, r := range res.Lookups {
		if r["lookup_type"] == "product_ndc" {
			t.Errorf("unexpected product_ndc lookup row: %v", r)
		}
	}

	// Marketing status maps to each application carrying it.
	marketing := func(term string) map[string]bool {
		m := map[string]bool{}
		for _, r := range res.Lookups {
			if r["lookup_type"] == "marketing_status" && r["term"] == term {
				m[r["appl_no_stripped"]] = true
			}
		}
		return m
	}
	if !marketing("Prescription")["125557"] {
		t.Errorf("Prescription marketing_status should include 125557")
	}
	if !marketing("Over-the-counter")["90123"] {
		t.Errorf("Over-the-counter marketing_status should include 90123")
	}
}

// --- submissions appl_type inheritance -------------------------------------------

func TestSubmissionsInheritApplType(t *testing.T) {
	res := extractFixtures(t)
	byStripped := make(map[string]string, len(res.Submissions))
	for _, r := range res.Submissions {
		byStripped[r["appl_no_stripped"]] = r["appl_type"]
	}
	for stripped, wantType := range map[string]string{
		"125557": "BLA",
		"75123":  "ANDA",
		"12345":  "NDA",
		"207500": "NDA",
		"90123":  "ANDA",
	} {
		if byStripped[stripped] != wantType {
			t.Errorf("submission %s appl_type = %q, want %q", stripped, byStripped[stripped], wantType)
		}
	}
}

// --- parsing ---------------------------------------------------------------------

func TestClassify(t *testing.T) {
	cases := []struct{ name, want string }{
		{"Products.txt", "products"},
		{"drugsfda_products.tsv", "products"},
		{"Product.txt", "products"},
		{"Applications.txt", "applications"},
		{"drugsfda_applications.tsv", "applications"},
		{"Submissions.txt", "submissions"},
		{"drugsfda_submissions.tsv", "submissions"},
		{"SubmissionPropertyType.txt", ""}, // sub-table rejected
		{"README.txt", ""},
	}
	for _, c := range cases {
		if got := Classify(c.name); got != c.want {
			t.Errorf("Classify(%q) = %q, want %q", c.name, got, c.want)
		}
	}
}

func TestParseTSVReaderToleratesRaggedAndBlank(t *testing.T) {
	in := "ApplNo\tApplType\tDrugName\n012345\tNDA\tFoo\n\n207500\tNDA\n"
	tbl, err := ParseTSVReader(strings.NewReader(in), "x.tsv")
	if err != nil {
		t.Fatalf("ParseTSVReader: %v", err)
	}
	if tbl.SourceName != "x.tsv" {
		t.Errorf("SourceName = %q, want x.tsv", tbl.SourceName)
	}
	if len(tbl.Header) != 3 {
		t.Fatalf("header = %v, want 3 columns", tbl.Header)
	}
	if len(tbl.Rows) != 2 {
		t.Fatalf("rows = %d, want 2 (blank line skipped)", len(tbl.Rows))
	}
	// Short second row reads missing fields as empty via the builders.
	rows, _ := BuildProducts(tbl)
	if len(rows) != 2 {
		t.Fatalf("BuildProducts rows = %d, want 2", len(rows))
	}
	if rows[1]["drug_name"] != "" {
		t.Errorf("short row drug_name = %q, want empty", rows[1]["drug_name"])
	}
	if rows[1]["appl_no_stripped"] != "207500" {
		t.Errorf("short row appl_no_stripped = %q, want 207500", rows[1]["appl_no_stripped"])
	}
}

func TestCombinedApplicationNumberColumn(t *testing.T) {
	// NDC-style combined ApplicationNumber + space-containing header (normKey is
	// space/underscore-insensitive), mirroring the Python combined-column test.
	in := "ProductNDC\tApplicationNumber\tProprietaryName\tNonProprietary Name\n" +
		"00000-001\tNDA012345\tExamplestatin\tExamplestatin\n" +
		"00000-002\tBLA125557\tKeytruda\tPembrolizumab\n"
	tbl, err := ParseTSVReader(strings.NewReader(in), "ndc_products.tsv")
	if err != nil {
		t.Fatalf("ParseTSVReader: %v", err)
	}
	rows, _ := BuildProducts(tbl)
	byRaw := make(map[string]Row, len(rows))
	for _, r := range rows {
		byRaw[r["appl_no_raw"]] = r
	}
	if byRaw["NDA012345"]["appl_type"] != "NDA" || byRaw["NDA012345"]["appl_no_stripped"] != "12345" {
		t.Errorf("NDA012345 row = %v", byRaw["NDA012345"])
	}
	if byRaw["NDA012345"]["product_ndc"] != "00000-001" {
		t.Errorf("product_ndc = %q, want 00000-001", byRaw["NDA012345"]["product_ndc"])
	}
	if byRaw["BLA125557"]["drug_name"] != "Keytruda" {
		t.Errorf("drug_name = %q, want Keytruda", byRaw["BLA125557"]["drug_name"])
	}
	if byRaw["BLA125557"]["active_ingredient"] != "Pembrolizumab" {
		t.Errorf("active_ingredient = %q, want Pembrolizumab", byRaw["BLA125557"]["active_ingredient"])
	}
}

// --- encoding sanitization (cp1252 fallback for invalid UTF-8 bytes) ------------

func TestToValidUTF8(t *testing.T) {
	cases := []struct{ name, in, want string }{
		{"ascii passthrough", "NDA012345", "NDA012345"},
		{"utf8 passthrough", "v\u00e1lido \u2713", "v\u00e1lido \u2713"},
		{"right single quote", "Men\x92s Rogaine", "Men\u2019s Rogaine"},
		{"en dash", "Approval \x96 March 23", "Approval \u2013 March 23"},
		{"latin1 byte", "caf\xe9", "caf\u00e9"},
		{"quotes and bullet", "\x93quoted\x94 \x95", "\u201cquoted\u201d \u2022"},
		{"undefined cp1252 byte -> U+FFFD", "a\x81b\x8dc\x8fd\x90e\x9df", "a\ufffdb\ufffdc\ufffdd\ufffde\ufffdf"},
		{"truncated multibyte lead", "trunc\xe2", "trunc\u00e2"},
		{"surrogate halves byte-by-byte", "x\xed\xa0\x80y", "x\u00ed\u00a0\u20acy"},
		{"empty", "", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := toValidUTF8(c.in); got != c.want {
				t.Errorf("toValidUTF8(%q) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}

func TestParseTSVReaderSanitizesCP1252(t *testing.T) {
	// Real-feed shape: \r\n line endings and cp1252 bytes in the notes column.
	in := "ApplNo\tSubmissionsPublicNotes\r\n021812\tLabel for Men\x92s Rogaine\r\n"
	tbl, err := ParseTSVReader(strings.NewReader(in), "Submissions.txt")
	if err != nil {
		t.Fatalf("ParseTSVReader: %v", err)
	}
	if len(tbl.Rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(tbl.Rows))
	}
	if got := tbl.Rows[0][1]; got != "Label for Men\u2019s Rogaine" {
		t.Errorf("notes cell = %q, want %q", got, "Label for Men\u2019s Rogaine")
	}
}

// TestParityGoldenTSVCP1252 extends the cross-language parity gate to dirty inputs: the
// golden was generated with the PYTHON reference (drugsfda_products._read_tsv +
// _build_submissions + schemas.write_tsv) from the same cp1252 fixture bytes, so a
// byte-for-byte match proves Go and Python sanitize identically on real-feed data.
func TestParityGoldenTSVCP1252(t *testing.T) {
	tbl, err := ParseTSV(filepath.Join("testdata", "dirty", "Submissions.txt"))
	if err != nil {
		t.Fatalf("ParseTSV(dirty/Submissions.txt): %v", err)
	}
	res := Extract(Tables{Submissions: &tbl})
	got := renderTSV(t, SubmissionsColumns, res.Submissions)
	want := readGolden(t, "drugsfda_submissions_cp1252.tsv")
	if !bytes.Equal(got, want) {
		t.Errorf("dirty submissions TSV differs from Python golden\n--- got ---\n%s\n--- want ---\n%s", got, want)
	}
}

// --- determinism + parity --------------------------------------------------------

func TestDeterminism(t *testing.T) {
	a := extractFixtures(t)
	b := extractFixtures(t)
	for _, c := range []struct {
		name    string
		columns []string
		ra, rb  []Row
	}{
		{"products", ProductsColumns, a.Products, b.Products},
		{"applications", ApplicationsColumns, a.Applications, b.Applications},
		{"submissions", SubmissionsColumns, a.Submissions, b.Submissions},
		{"lookups", LookupsColumns, a.Lookups, b.Lookups},
	} {
		if !bytes.Equal(renderTSV(t, c.columns, c.ra), renderTSV(t, c.columns, c.rb)) {
			t.Errorf("%s output is not deterministic across runs", c.name)
		}
	}
}

// TestParityGoldenTSV is the cross-language parity gate: the Go TSV output must be
// byte-for-byte identical to the Python reference output (golden fixtures) for all four
// tables, given byte-identical inputs.
func TestParityGoldenTSV(t *testing.T) {
	res := extractFixtures(t)
	for _, c := range []struct {
		golden  string
		columns []string
		rows    []Row
	}{
		{"drugsfda_products.tsv", ProductsColumns, res.Products},
		{"drugsfda_applications.tsv", ApplicationsColumns, res.Applications},
		{"drugsfda_submissions.tsv", SubmissionsColumns, res.Submissions},
		{"drugsfda_lookups.tsv", LookupsColumns, res.Lookups},
	} {
		got := renderTSV(t, c.columns, c.rows)
		want := readGolden(t, c.golden)
		if !bytes.Equal(got, want) {
			t.Errorf("%s: TSV bytes differ from Python golden\n--- got ---\n%s\n--- want ---\n%s", c.golden, got, want)
		}
	}
}

// TestExtractMissingTablesWarns verifies absent tables produce deterministic warnings and
// no rows (mirrors the Python "no X table found in inputs" provenance record).
func TestExtractMissingTablesWarns(t *testing.T) {
	products := loadTable(t, "drugsfda_products.tsv")
	res := Extract(Tables{Products: &products})
	if !res.HaveProducts || res.HaveApplications || res.HaveSubmissions {
		t.Fatalf("presence flags = (%v,%v,%v), want (true,false,false)", res.HaveProducts, res.HaveApplications, res.HaveSubmissions)
	}
	if len(res.Applications) != 0 || len(res.Submissions) != 0 {
		t.Errorf("absent tables should yield zero rows")
	}
	tables := map[string]bool{}
	for _, w := range res.Warnings {
		tables[w.Table] = true
	}
	if !tables["applications"] || !tables["submissions"] {
		t.Errorf("warnings should note missing applications+submissions, got %v", res.Warnings)
	}
	// Lookups still derive from products.
	if len(res.Lookups) == 0 {
		t.Errorf("lookups should be built from products even without applications/submissions")
	}
}
