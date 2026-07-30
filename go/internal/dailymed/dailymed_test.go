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
	tables, err := Extract(context.Background(), []string{mockFixture}, limit)
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
	if _, err := Extract(context.Background(), []string{"testdata/does_not_exist.xml.gz"}, 0); err == nil {
		t.Error("Extract should error on a missing input file")
	}
}

// regenerateGoldens (documentation only): the testdata/golden/*.tsv files were produced by
// running the Python reference extractor on the identical fixture and rendering each
// interim parquet through polars write_csv(separator="\t"):
//
//	refs = spl_xml.extract([ArtifactRef(uri=fixture, blake3=hash_file(fixture), ...)], ctx)
//	pl.read_parquet(ref.uri).write_csv(out, separator="\t")
//
// Re-run that (uv run python ...) to refresh the goldens if the Python contract changes.
