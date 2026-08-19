package dailymed

import (
	"bytes"
	"compress/gzip"
	"context"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"testing"

	"github.com/glusman-team/dakp/go/internal/pipeline"
)

// fixtureSourceID is the BLAKE3 of testdata/dailymed_spl.xml.gz — byte-identical to the
// Python test fixture (tests/fixtures/pipeline/dailymed/dailymed_spl.xml.gz), so it is the
// source artifact id that underlies every source_record_id in the goldens.
const fixtureSourceID = "b3:9ea784ce2eed27f988d8d243aedaeb864ccd5dce02d548246ff4e4ae71e8db11"

const mockFixture = "testdata/dailymed_spl.xml.gz"

var b3IDRe = regexp.MustCompile(`^b3:[0-9a-f]{64}$`)

// extractFixture runs the bounded-parallel extractor over the mock fixture.
func extractFixture(t *testing.T, limit int) *Tables {
	t.Helper()
	tables, err := Extract(context.Background(), []string{mockFixture}, nil, limit)
	if err != nil {
		t.Fatalf("Extract(%s): %v", mockFixture, err)
	}
	return tables
}

// --- streaming parse of the mock fixture ---------------------------------------

func TestParseFileMockFixture(t *testing.T) {
	docs, err := ParseFile(mockFixture)
	if err != nil {
		t.Fatalf("ParseFile: %v", err)
	}
	if len(docs) != 3 {
		t.Fatalf("got %d documents, want 3", len(docs))
	}

	wantSetIDs := []string{"SETID-EXAMPLESTATIN-001", "SETID-IBUPROFEN-002", "SETID-OMEPRAZOLE-003"}
	for i, want := range wantSetIDs {
		if docs[i].SetID != want {
			t.Errorf("doc[%d].SetID = %q, want %q", i, docs[i].SetID, want)
		}
	}

	// Document 1: one NDA approval, one active + one inactive ingredient, two sections.
	d0 := docs[0]
	if !reflect.DeepEqual(d0.Approvals, []Approval{{ID: "012345", Code: "012345", Type: "NDA"}}) {
		t.Errorf("doc0.Approvals = %+v", d0.Approvals)
	}
	wantIng := []Ingredient{
		{Name: "Examplestatin", UNII: "UNII:QFX8B1R4QF", Role: "active"},
		{Name: "Lactose", UNII: "UNII:J2B2A4N98G", Role: "inactive"},
	}
	if !reflect.DeepEqual(d0.Ingredients, wantIng) {
		t.Errorf("doc0.Ingredients = %+v, want %+v", d0.Ingredients, wantIng)
	}
	if len(d0.Sections) != 2 {
		t.Fatalf("doc0 has %d sections, want 2", len(d0.Sections))
	}
	if d0.Sections[0].LOINC != "34067-9" || d0.Sections[0].Name != "INDICATIONS AND USAGE" {
		t.Errorf("doc0.Sections[0] = %+v", d0.Sections[0])
	}
	// raw_text preserves the (stripped) original; clean_text is whitespace-collapsed. For
	// this single-line fixture they coincide.
	wantText := "Examplestatin is indicated for the treatment of hypercholesterolemia and to reduce elevated LDL cholesterol in adults."
	if d0.Sections[0].RawText != wantText || d0.Sections[0].CleanText != wantText {
		t.Errorf("doc0.Sections[0] text = raw %q / clean %q, want %q", d0.Sections[0].RawText, d0.Sections[0].CleanText, wantText)
	}
	// Title falls back to the section name when there is no <title> element (mock shape).
	if d0.Sections[0].Title != "INDICATIONS AND USAGE" {
		t.Errorf("doc0.Sections[0].Title = %q", d0.Sections[0].Title)
	}

	// Document 3: the no-LOINC "HOW SUPPLIED" section is preserved with an empty code and a
	// parse warning (no data loss).
	d2 := docs[2]
	if len(d2.Sections) != 2 {
		t.Fatalf("doc2 has %d sections, want 2", len(d2.Sections))
	}
	noLoinc := d2.Sections[1]
	if noLoinc.LOINC != "" || noLoinc.Name != "HOW SUPPLIED" || noLoinc.Title != "HOW SUPPLIED" {
		t.Errorf("doc2 no-LOINC section = %+v", noLoinc)
	}
	if len(d2.Warnings) == 0 {
		t.Error("doc2 should carry a 'section missing LOINC code' warning")
	}
}

// --- normalized tables: row counts + column contracts --------------------------

func TestExtractRowCountsAndColumns(t *testing.T) {
	tables := extractFixture(t, 0)

	if len(tables.Documents) != 6 {
		t.Errorf("documents rows = %d, want 6 (3 docs x 2 sections)", len(tables.Documents))
	}
	if len(tables.Sets) != 3 {
		t.Errorf("sets rows = %d, want 3", len(tables.Sets))
	}
	if len(tables.Approvals) != 3 {
		t.Errorf("approvals rows = %d, want 3", len(tables.Approvals))
	}
	if len(tables.Ingredients) != 4 {
		t.Errorf("ingredients rows = %d, want 4 (1 active each + 1 inactive on doc 1)", len(tables.Ingredients))
	}
	if len(tables.Sections) != 6 {
		t.Errorf("sections rows = %d, want 6", len(tables.Sections))
	}
	if tables.Warnings < 1 {
		t.Errorf("warnings = %d, want >= 1 (the no-LOINC section)", tables.Warnings)
	}

	// Column contracts match the Python SPL_*_COLUMNS / DAILYMED_SPL_DOCUMENTS_COLUMNS.
	for _, tf := range tables.TableFiles() {
		if len(tf.Rows) > 0 && len(tf.Rows[0]) != len(tf.Columns) {
			t.Errorf("%s: row width %d != column count %d", tf.Name, len(tf.Rows[0]), len(tf.Columns))
		}
	}
	if got := tables.TableFiles()[0].Columns; !reflect.DeepEqual(got, DocumentsColumns) {
		t.Errorf("documents columns = %v", got)
	}
}

// --- cross-language parity: byte-for-byte against the Python/polars goldens -----

// TestGoldenTSVParity is the headline parity check: for the fixture, every Go TSV table
// must be byte-identical to the golden produced by the Python reference extractor
// (spl_xml.extract) rendered through polars write_csv(separator="\t"). The goldens under
// testdata/golden/ were generated by that Python path (see regenerateGoldens note below),
// so this locks column order, column names, every value, the source_record_id derivation,
// AND the polars TSV quoting/line-ending behavior (including "" for empty strings).
func TestGoldenTSVParity(t *testing.T) {
	tables := extractFixture(t, 0)
	for _, tf := range tables.TableFiles() {
		goldenPath := filepath.Join("testdata", "golden", tf.Name)
		want, err := os.ReadFile(goldenPath)
		if err != nil {
			t.Fatalf("read golden %s: %v", goldenPath, err)
		}
		if got := RenderTSV(tf.Columns, tf.Rows); got != string(want) {
			t.Errorf("%s: Go TSV != Python/polars golden\n--- got ---\n%s\n--- want ---\n%s", tf.Name, got, want)
		}
	}
}

// TestSourceRecordIDDerivationMatchesGolden ties the golden's ids to the shared derivation:
// the first section id in the golden must equal pipeline.SourceRecordID (which is itself
// parity-locked to spl_xml._source_record_id in internal/pipeline). This proves the golden
// and the Go code agree on HOW ids are derived, not just on their literal bytes.
func TestSourceRecordIDDerivationMatchesGolden(t *testing.T) {
	got := pipeline.SourceRecordID(fixtureSourceID, "section", "SETID-EXAMPLESTATIN-001", "34067-9")
	want := "b3:f3cbfffe07fe5b0bed219ae86dc94a905271aa25e8b49df4331da2560c3558ff" // golden row 1
	if got != want {
		t.Errorf("SourceRecordID = %q, want golden %q", got, want)
	}

	tables := extractFixture(t, 0)
	if len(tables.InputIDs) != 1 || tables.InputIDs[0] != fixtureSourceID {
		t.Errorf("InputIDs = %v, want [%s]", tables.InputIDs, fixtureSourceID)
	}
}

// --- determinism + b3 format ---------------------------------------------------

func TestSourceRecordIDFormatAndUniqueness(t *testing.T) {
	tables := extractFixture(t, 0)
	// source_record_id is column 0 of sets/approvals/ingredients/sections.
	for _, tf := range tables.TableFiles()[1:] {
		seen := map[string]bool{}
		for _, row := range tf.Rows {
			id := row[0]
			if !b3IDRe.MatchString(id) {
				t.Errorf("%s: source_record_id %q is not a b3:<64hex> id", tf.Name, id)
			}
			if seen[id] {
				t.Errorf("%s: duplicate source_record_id %q", tf.Name, id)
			}
			seen[id] = true
		}
	}
}

func TestExtractIsDeterministicAcrossRunsAndConcurrency(t *testing.T) {
	seq := extractFixture(t, 1) // sequential
	par := extractFixture(t, 8) // bounded-parallel
	again := extractFixture(t, 1)

	if !reflect.DeepEqual(seq, par) {
		t.Error("parallel extraction differs from sequential (output must be order-independent)")
	}
	if !reflect.DeepEqual(seq, again) {
		t.Error("re-extraction is not byte-stable (source_record_id must be deterministic)")
	}
}

// --- TSV quoting: locked against polars 1.43 ------------------------------------

// TestTSVFieldQuotingMatchesPolars pins the field-quoting rule to polars write_csv output
// (probe: empty -> "" , plain/leading-space -> unquoted, tab/newline/CR/quote -> quoted
// with doubled inner quotes). The fixture only exercises the empty-string case, so this
// covers the rest explicitly.
func TestTSVFieldQuotingMatchesPolars(t *testing.T) {
	cases := []struct{ in, want string }{
		{"", "\"\""},
		{"abc", "abc"},
		{" ", " "},
		{" abc", " abc"},
		{"a\tb", "\"a\tb\""},
		{"a\nb", "\"a\nb\""},
		{"a\rb", "\"a\rb\""},
		{"a\"b", "\"a\"\"b\""},
		{"012345", "012345"},
	}
	for _, c := range cases {
		if got := tsvField(c.in); got != c.want {
			t.Errorf("tsvField(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// --- gzip-awareness: plain and gzipped content agree ---------------------------

func TestGzipAndPlainAgree(t *testing.T) {
	gzBytes, err := os.ReadFile(mockFixture)
	if err != nil {
		t.Fatal(err)
	}
	zr, err := gzip.NewReader(bytes.NewReader(gzBytes))
	if err != nil {
		t.Fatal(err)
	}
	var plain bytes.Buffer
	if _, err := plain.ReadFrom(zr); err != nil {
		t.Fatal(err)
	}
	plainPath := filepath.Join(t.TempDir(), "dailymed_spl.xml")
	if err := os.WriteFile(plainPath, plain.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}

	fromGZ, err := ParseFile(mockFixture)
	if err != nil {
		t.Fatal(err)
	}
	fromPlain, err := ParseFile(plainPath)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(fromGZ, fromPlain) {
		t.Error("gzipped and plain parses disagree")
	}

	// The mock fixture is namespace-free in both forms.
	if isHL7v3, err := detectHL7v3(mockFixture); err != nil || isHL7v3 {
		t.Errorf("detectHL7v3(mock) = %v, %v; want false, nil", isHL7v3, err)
	}
}

// --- HL7 v3 (real DailyMed) shape ----------------------------------------------

func TestParseHL7v3Fixture(t *testing.T) {
	const path = "testdata/hl7v3_spl.xml.gz"
	if isHL7v3, err := detectHL7v3(path); err != nil || !isHL7v3 {
		t.Fatalf("detectHL7v3(hl7v3) = %v, %v; want true, nil", isHL7v3, err)
	}
	docs, err := ParseFile(path)
	if err != nil {
		t.Fatalf("ParseFile: %v", err)
	}
	// Only the <document> is parsed; the section outside any document is ignored.
	if len(docs) != 1 {
		t.Fatalf("got %d documents, want 1 (outside-document section must be ignored)", len(docs))
	}
	d := docs[0]

	// Set id comes from <setId root=>, lowercased.
	if want := "abcdef12-3456-7890-abcd-ef1234567890"; d.SetID != want {
		t.Errorf("SetID = %q, want %q", d.SetID, want)
	}
	// Approval: NDA id from id[@root=NDA OID]/@extension, type from code[@codeSystem]/@code.
	if !reflect.DeepEqual(d.Approvals, []Approval{{ID: "012345", Code: "012345", Type: "NDA"}}) {
		t.Errorf("Approvals = %+v", d.Approvals)
	}
	// Ingredients: active from the nested activeMoiety, inactive from
	// inactiveIngredientSubstance; each with UNII + role, deduplicated.
	wantIng := []Ingredient{
		{Name: "Aspirin", UNII: "UNII:R16CO5Y76E", Role: "active"},
		{Name: "Microcrystalline Cellulose", UNII: "UNII:OP1R32D61U", Role: "inactive"},
	}
	if !reflect.DeepEqual(d.Ingredients, wantIng) {
		t.Errorf("Ingredients = %+v, want %+v", d.Ingredients, wantIng)
	}
	// Sections: LOINC from the section's <code code=>, name falls back to the SECTION_CODE_NAMES
	// map (no name attr in HL7 v3), title from <title>. Section text includes the title text
	// (itertext over the whole section), whitespace-collapsed in clean_text.
	if len(d.Sections) != 2 {
		t.Fatalf("got %d sections, want 2", len(d.Sections))
	}
	s0, s1 := d.Sections[0], d.Sections[1]
	if s0.LOINC != "34067-9" || s0.Name != "indications_and_usage" || s0.Title != "INDICATIONS AND USAGE" {
		t.Errorf("section0 = %+v", s0)
	}
	if want := "INDICATIONS AND USAGE Aspirin is indicated for pain."; s0.CleanText != want {
		t.Errorf("section0.CleanText = %q, want %q", s0.CleanText, want)
	}
	if !strings.Contains(s0.RawText, "Aspirin   is indicated for   pain.") {
		t.Errorf("section0.RawText should preserve original internal whitespace, got %q", s0.RawText)
	}
	if s1.LOINC != "34070-3" || s1.Name != "contraindications" || s1.Title != "CONTRAINDICATIONS" {
		t.Errorf("section1 = %+v", s1)
	}
	if want := "CONTRAINDICATIONS None known."; s1.CleanText != want {
		t.Errorf("section1.CleanText = %q, want %q", s1.CleanText, want)
	}
}

// --- input discovery + output --------------------------------------------------

func TestListSPLFilesSortsAndFilters(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"b.xml.gz", "a.xml", "notes.txt", "c.XML.GZ"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Mkdir(filepath.Join(dir, "sub.xml"), 0o755); err != nil { // directory, must be skipped
		t.Fatal(err)
	}
	got, err := ListSPLFiles(dir)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		filepath.Join(dir, "a.xml"),
		filepath.Join(dir, "b.xml.gz"),
		filepath.Join(dir, "c.XML.GZ"),
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ListSPLFiles = %v, want %v", got, want)
	}

	// A single SPL file path is returned as-is; a non-SPL file errors.
	if got, err := ListSPLFiles(filepath.Join(dir, "a.xml")); err != nil || len(got) != 1 {
		t.Errorf("ListSPLFiles(file) = %v, %v", got, err)
	}
	if _, err := ListSPLFiles(filepath.Join(dir, "notes.txt")); err == nil {
		t.Error("ListSPLFiles(non-SPL file) should error")
	}
}

func TestWriteDirEmitsAllTables(t *testing.T) {
	tables := extractFixture(t, 0)
	out := t.TempDir()
	if err := tables.WriteDir(out); err != nil {
		t.Fatalf("WriteDir: %v", err)
	}
	for _, tf := range tables.TableFiles() {
		path := filepath.Join(out, tf.Name)
		data, err := os.ReadFile(path)
		if err != nil {
			t.Errorf("missing output %s: %v", tf.Name, err)
			continue
		}
		if got, want := string(data), RenderTSV(tf.Columns, tf.Rows); got != want {
			t.Errorf("%s on disk != rendered TSV", tf.Name)
		}
	}
}

func TestExtractErrorsOnMissingFile(t *testing.T) {
	if _, err := Extract(context.Background(), []string{"testdata/does_not_exist.xml.gz"}, nil, 0); err == nil {
		t.Error("Extract should error on a missing input file")
	}
}

func TestExtractUsesProvidedSourceIDs(t *testing.T) {
	// A provided id is used verbatim — the input file is never re-hashed.
	provided := "b3:0000000000000000000000000000000000000000000000000000000000000000"
	tables, err := Extract(context.Background(), []string{mockFixture}, []string{provided}, 4)
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if len(tables.InputIDs) != 1 || tables.InputIDs[0] != provided {
		t.Errorf("InputIDs = %v, want [%q]", tables.InputIDs, provided)
	}

	// An empty id falls back to hashing the file.
	fallback, err := Extract(context.Background(), []string{mockFixture}, []string{""}, 4)
	if err != nil {
		t.Fatalf("Extract fallback: %v", err)
	}
	if len(fallback.InputIDs) != 1 || fallback.InputIDs[0] != fixtureSourceID {
		t.Errorf("fallback InputIDs = %v, want [%q]", fallback.InputIDs, fixtureSourceID)
	}

	// The provided id flows into the source_record_ids (sets carry the source id).
	if len(tables.Sets) == 0 || len(fallback.Sets) == 0 {
		t.Fatal("fixture should produce set rows")
	}
	if tables.Sets[0][0] == fallback.Sets[0][0] {
		t.Error("provided id did not change the source_record_id")
	}
}

// --- whitespace parity (Python str.split/strip vs Go unicode.IsSpace) ----------

// TestWhitespaceMatchesPython locks collapseWS to Python's str.split() whitespace class,
// which (unlike Go's unicode.IsSpace) treats U+001C..U+001F as whitespace. The golden
// fixture has none of these, so this pins the parity explicitly.
func TestWhitespaceMatchesPython(t *testing.T) {
	cases := []struct{ in, want string }{
		{"a\x1cb", "a b"},
		{"\x1c\x1d hello \x1e\x1f", "hello"},
		{"a\u0085b\u00a0c", "a b c"},
		{"plain text", "plain text"},
		{"  spaced  ", "spaced"},
	}
	for _, c := range cases {
		if got := collapseWS(c.in); got != c.want {
			t.Errorf("collapseWS(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// Note: U+001C..U+001F are illegal characters in XML 1.0, so they cannot reach the parser
// via well-formed SPL; the parity above is asserted at the helper level (not through an
// XML fixture) because encoding/xml rightly rejects them.

// --- HL7 v3 branch coverage (not exercised by the mock byte-golden) -------------

// parseHL7v3String parses an inline HL7 v3 fragment (wrapped in a namespaced <SPL> root)
// through the streaming parser, for surgical branch tests.
func parseHL7v3String(t *testing.T, body string) []DocumentRecord {
	t.Helper()
	doc := `<?xml version="1.0" encoding="UTF-8"?>` + "\n" + `<SPL xmlns="urn:hl7-org:v3">` + body + `</SPL>`
	docs, err := parseDocuments(strings.NewReader(doc), true)
	if err != nil {
		t.Fatalf("parseDocuments: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("got %d documents, want 1", len(docs))
	}
	return docs
}

func TestHL7v3Branches(t *testing.T) {
	const nda = `root="2.16.840.1.113883.3.150"`
	const appl = `codeSystem="2.16.840.1.113883.3.26.1.1"`

	t.Run("duplicate approval last type wins", func(t *testing.T) {
		docs := parseHL7v3String(t, `<document><setId root="S"/>`+
			`<subjectOf><approval><id `+nda+` extension="NDA1"/><code code="NDA" `+appl+`/></approval></subjectOf>`+
			`<subjectOf><approval><id `+nda+` extension="NDA1"/><code code="ANDA" `+appl+`/></approval></subjectOf>`+
			`</document>`)
		ap := docs[0].Approvals
		if len(ap) != 1 || ap[0].ID != "NDA1" || ap[0].Type != "ANDA" {
			t.Errorf("approvals = %+v, want one NDA1 with last-wins type ANDA", ap)
		}
	})

	t.Run("duplicate approval empty type overwrites", func(t *testing.T) {
		// Legacy quirk: type_by_id is overwritten unconditionally, so a later approval with
		// the same NDA id but no application-type code resets the returned type to "".
		docs := parseHL7v3String(t, `<document><setId root="S"/>`+
			`<subjectOf><approval><id `+nda+` extension="NDA1"/><code code="NDA" `+appl+`/></approval></subjectOf>`+
			`<subjectOf><approval><id `+nda+` extension="NDA1"/></approval></subjectOf>`+
			`</document>`)
		ap := docs[0].Approvals
		if len(ap) != 1 || ap[0].Type != "" {
			t.Errorf("approvals = %+v, want one NDA1 with type overwritten to empty", ap)
		}
	})

	t.Run("inactive via ingredient classCode IACT", func(t *testing.T) {
		docs := parseHL7v3String(t, `<document><setId root="S"/>`+
			`<ingredient classCode="IACT"><ingredientSubstance>`+
			`<name>Water</name><code code="059QF0KO0R"/>`+
			`</ingredientSubstance></ingredient></document>`)
		want := []Ingredient{{Name: "Water", UNII: "UNII:059QF0KO0R", Role: "inactive"}}
		if !reflect.DeepEqual(docs[0].Ingredients, want) {
			t.Errorf("ingredients = %+v, want %+v", docs[0].Ingredients, want)
		}
	})

	t.Run("section code prefers the LOINC codeSystem OID", func(t *testing.T) {
		// A code tagged with the LOINC codeSystem OID beats an earlier LOINC-shaped
		// code from another system (mirrors spl_xml._collect_sections).
		docs := parseHL7v3String(t, `<document><setId root="S"/>`+
			`<section><code code="99999-9" codeSystem="9.9.9.9.9"/>`+
			`<code code="34066-1" codeSystem="2.16.840.1.113883.6.1"/>`+
			`<title>BOXED WARNING</title><text>Do not use.</text></section>`+
			`<section><code code="43685-7" codeSystem="2.16.840.1.113883.6.1"/>`+
			`<title>WARNINGS AND PRECAUTIONS</title><text>Not recommended.</text></section>`+
			`<section><code code="42229-5" codeSystem="2.16.840.1.113883.6.1"/>`+
			`<title>SPL UNCLASSIFIED SECTION</title><text>Manufactured by X.</text></section>`+
			`</document>`)
		secs := docs[0].Sections
		if len(secs) != 3 {
			t.Fatalf("got %d sections, want 3", len(secs))
		}
		if secs[0].LOINC != "34066-1" || secs[0].Name != "boxed_warning" {
			t.Errorf("section0 = %+v, want LOINC 34066-1 boxed_warning", secs[0])
		}
		if secs[1].LOINC != "43685-7" || secs[1].Name != "warnings_and_precautions" {
			t.Errorf("section1 = %+v, want LOINC 43685-7 warnings_and_precautions", secs[1])
		}
		// 42229-5 is SPL UNCLASSIFIED, never warnings content (historical mislabel).
		if secs[2].LOINC != "42229-5" || secs[2].Name != "spl_unclassified" {
			t.Errorf("section2 = %+v, want LOINC 42229-5 spl_unclassified", secs[2])
		}
	})

	t.Run("active via activeIngredientSubstance", func(t *testing.T) {
		docs := parseHL7v3String(t, `<document><setId root="S"/>`+
			`<activeMoiety><activeIngredientSubstance>`+
			`<name>Foo</name><code code="FOO1"/>`+
			`</activeIngredientSubstance></activeMoiety></document>`)
		want := []Ingredient{{Name: "Foo", UNII: "UNII:FOO1", Role: "active"}}
		if !reflect.DeepEqual(docs[0].Ingredients, want) {
			t.Errorf("ingredients = %+v, want %+v", docs[0].Ingredients, want)
		}
	})

	t.Run("ingredient dedup by role+unii+lower name", func(t *testing.T) {
		docs := parseHL7v3String(t, `<document><setId root="S"/>`+
			`<inactiveIngredientSubstance><name>Dup</name><code code="D1"/></inactiveIngredientSubstance>`+
			`<inactiveIngredientSubstance><name>dup</name><code code="D1"/></inactiveIngredientSubstance>`+
			`</document>`)
		if len(docs[0].Ingredients) != 1 {
			t.Errorf("ingredients = %+v, want dedup to 1 (case-insensitive name)", docs[0].Ingredients)
		}
	})

	t.Run("nested sections both emit and parent includes child text", func(t *testing.T) {
		docs := parseHL7v3String(t, `<document><setId root="S"/>`+
			`<component><section> `+
			`<code code="34067-9"/> <title>PARENT</title> <text>Parent text.</text> `+
			`<component><section> `+
			`<code code="34070-3"/> <title>CHILD</title> <text>Child text.</text> `+
			`</section></component> `+
			`</section></component></document>`)
		secs := docs[0].Sections
		if len(secs) != 2 {
			t.Fatalf("sections = %d, want 2 (parent + nested child)", len(secs))
		}
		if secs[0].LOINC != "34067-9" || !strings.Contains(secs[0].CleanText, "Parent text.") || !strings.Contains(secs[0].CleanText, "Child text.") {
			t.Errorf("parent section = %+v; clean_text must include nested child text", secs[0])
		}
		if secs[1].LOINC != "34070-3" || secs[1].CleanText != "CHILD Child text." {
			t.Errorf("child section = %+v", secs[1])
		}
	})
}

// regenerateGoldens (documentation only): the testdata/golden/*.tsv files were produced by
// running the Python reference extractor on the identical fixture and rendering each
// interim parquet through polars write_csv(separator="\t"):
//
//	refs = spl_xml.extract([ArtifactRef(uri=fixture, blake3=hash_file(fixture), ...)], ctx)
//	pl.read_parquet(ref.uri).write_csv(out, separator="\t")
//
// Re-run that (uv run python ...) to refresh the goldens if the Python contract changes.
