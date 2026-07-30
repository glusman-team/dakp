package faers

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/glusman-team/dakp/go/internal/blake3store"
)

// goldenCasesTSV is the exact faers_cases.tsv emitted by the Python reference
// (src/dakp_pipeline/extract/faers_ascii.py) for the testdata/ fixtures, captured from a
// real run. It is the cross-language parity oracle: the Go extractor must reproduce it
// byte-for-byte. (faers_cases.tsv excludes source_record_id, so it is hash-independent.)
const goldenCasesTSV = "quarter\tprimaryid\tcaseid\tsource\toccp_cod\treporter_country\tdrugname\tingredient\tnda\tindication\teffects\n" +
	"24Q3\t1001\t5001\tPERIODIC\tMD\tUS\tExamplestatin\tExamplestatin\t12345\thypercholesterolemia\tmyalgia$rhabdomyolysis\n" +
	"24Q3\t1002\t5002\tDIRECT\tPH\tGB\tAdvil\tIbuprofen\t17977\theadache\tnausea\n" +
	"24Q2\t2002\t5003\tDIRECT\tOT\tCN\tAspirin\tAcetylsalicylic acid\t20000\tarthritis\tdyspepsia\n"

// --- test helpers --------------------------------------------------------------

// testSource builds a Source from a testdata file, hashing its content like the CLI does.
func testSource(t *testing.T, name string) Source {
	t.Helper()
	content, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read testdata %s: %v", name, err)
	}
	family, quarter := FamilyAndQuarter(name)
	return Source{Quarter: quarter, Family: family, Content: content, SourceName: name, SourceB3: blake3store.HashBytes(content)}
}

// loadTestdata loads every FAERS .txt under testdata/ as Sources (sorted by name).
func loadTestdata(t *testing.T) []Source {
	t.Helper()
	entries, err := os.ReadDir("testdata")
	if err != nil {
		t.Fatalf("read testdata: %v", err)
	}
	var names []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".txt") {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	var srcs []Source
	for _, n := range names {
		family, quarter := FamilyAndQuarter(n)
		if family == "" || quarter == "" {
			continue
		}
		srcs = append(srcs, testSource(t, n))
	}
	return srcs
}

// parseFamilies parses the named testdata files into a family -> Table map (one quarter).
func parseFamilies(t *testing.T, names ...string) map[string]*Table {
	t.Helper()
	warn := &Warnings{}
	fam := map[string]*Table{}
	for _, n := range names {
		src := testSource(t, n)
		if tbl := ParseSource(src, warn); tbl != nil {
			fam[src.Family] = tbl
		}
	}
	return fam
}

func inlineSource(quarter, family, name, content string) Source {
	b := []byte(content)
	return Source{Quarter: quarter, Family: family, Content: b, SourceName: name, SourceB3: blake3store.HashBytes(b)}
}

func byPrimaryID(cases []Case) map[string]Case {
	out := make(map[string]Case, len(cases))
	for _, c := range cases {
		out[c.PrimaryID] = c
	}
	return out
}

// --- FamilyAndQuarter ----------------------------------------------------------

func TestFamilyAndQuarter(t *testing.T) {
	cases := []struct {
		name           string
		wantFam, wantQ string
	}{
		{"DEMO24Q3.txt", "DEMO", "24Q3"},
		{"ascii/DRUG24Q2.txt", "DRUG", "24Q2"},
		{"delete24q3.txt", "DELETE", "24Q3"}, // lowercase + lowercase q
		{"RPSR12Q1.TXT", "RPSR", "12Q1"},
		{"README.txt", "", ""},         // no family, no quarter
		{"DEMO.txt", "DEMO", ""},       // family but no quarter
		{"notes_24Q3.log", "", "24Q3"}, // quarter but no family
	}
	for _, c := range cases {
		fam, q := FamilyAndQuarter(c.name)
		if fam != c.wantFam || q != c.wantQ {
			t.Errorf("FamilyAndQuarter(%q) = (%q,%q), want (%q,%q)", c.name, fam, q, c.wantFam, c.wantQ)
		}
	}
}

// --- per-file parsing ----------------------------------------------------------

func TestParseLowercasesHeadersAndDropsTrailingEmptyColumn(t *testing.T) {
	src := inlineSource("24Q3", "DEMO", "DEMO24Q3.txt", "PRIMARYID$CASEID$\r\n1001$5001$\r\n")
	tbl := ParseSource(src, &Warnings{})
	if tbl == nil {
		t.Fatal("expected a table")
	}
	// Provenance first, then lowercased file columns; no trailing empty column.
	want := []string{"quarter", "source_file", "source_record_id", "primaryid", "caseid"}
	if !reflectEqual(tbl.Columns, want) {
		t.Fatalf("columns = %v, want %v", tbl.Columns, want)
	}
	if got := tbl.Rows[0][indexOf(tbl.Columns, "primaryid")]; got != "1001" {
		t.Fatalf("primaryid = %q, want 1001", got)
	}
}

func TestParseResolvesLegacyISRAsPrimaryID(t *testing.T) {
	// Pre-2014 FAERS used ISR instead of PRIMARYID.
	src := inlineSource("12Q1", "DEMO", "DEMO12Q1.txt", "ISR$CASEID$\r\n9001$6001$\r\n")
	tbl := ParseSource(src, &Warnings{})
	if tbl == nil {
		t.Fatal("expected a table")
	}
	if !containsString(tbl.Columns, "primaryid") || containsString(tbl.Columns, "isr") {
		t.Fatalf("isr not resolved to primaryid: %v", tbl.Columns)
	}
	if got := tbl.Rows[0][indexOf(tbl.Columns, "primaryid")]; got != "9001" {
		t.Fatalf("primaryid = %q, want 9001", got)
	}
}

func TestParseHandlesCRLFAndTrailingDollarPreservingNDALeadingZeroes(t *testing.T) {
	tbl := parseFamilies(t, "DRUG24Q3.txt")["DRUG"]
	if tbl == nil {
		t.Fatal("expected DRUG table")
	}
	if len(tbl.Rows) != 3 {
		t.Fatalf("rows = %d, want 3", len(tbl.Rows))
	}
	nda := indexOf(tbl.Columns, "nda_num")
	var got []string
	for _, r := range tbl.Rows {
		got = append(got, cell(r, nda))
	}
	want := []string{"012345", "017977", "099999"} // all-Utf8: leading zeroes preserved
	if !reflectEqual(got, want) {
		t.Fatalf("nda_num = %v, want %v", got, want)
	}
}

func TestParseMissingPrimaryIDRecordsWarning(t *testing.T) {
	warn := &Warnings{}
	src := inlineSource("24Q3", "DEMO", "DEMO24Q3.txt", "CASEID$\r\n5001$\r\n")
	if tbl := ParseSource(src, warn); tbl != nil {
		t.Fatal("expected nil table for missing primaryid")
	}
	if !warn.Has("missing_primaryid") {
		t.Fatalf("expected missing_primaryid warning, got %v", warn.Items())
	}
}

func TestParseEmptyRecordsWarning(t *testing.T) {
	warn := &Warnings{}
	if tbl := ParseSource(inlineSource("24Q3", "DEMO", "DEMO24Q3.txt", "   \r\n"), warn); tbl != nil {
		t.Fatal("expected nil table for blank content")
	}
	if !warn.Has("empty_file") {
		t.Fatalf("expected empty_file warning, got %v", warn.Items())
	}
	// Header but no data rows.
	warn2 := &Warnings{}
	if tbl := ParseSource(inlineSource("24Q3", "DEMO", "DEMO24Q3.txt", "PRIMARYID$CASEID$\r\n"), warn2); tbl != nil {
		t.Fatal("expected nil table for header-only content")
	}
	if !warn2.Has("empty_file") {
		t.Fatalf("expected empty_file warning, got %v", warn2.Items())
	}
}

func TestParseTrimsFieldWhitespace(t *testing.T) {
	src := inlineSource("24Q3", "RPSR", "RPSR24Q3.txt", "PRIMARYID$RPSR_COD$\r\n1001$PERIODIC $\r\n")
	tbl := ParseSource(src, &Warnings{})
	if tbl == nil {
		t.Fatal("expected a table")
	}
	if got := tbl.Rows[0][indexOf(tbl.Columns, "rpsr_cod")]; got != "PERIODIC" {
		t.Fatalf("rpsr_cod = %q, want trimmed PERIODIC", got)
	}
}

// --- normalized source_record_id (b3-derived format) ---------------------------

var b3PrefixRe = regexp.MustCompile(`^[0-9a-f]{12}$`)

func TestParseSourceRecordIDIsB3Derived(t *testing.T) {
	// Golden per-file source_record_ids captured from the Python reference run on the
	// byte-identical fixtures (prefix = first 12 hex of the file's b3 content hash).
	golden := map[string][]string{
		"DRUG24Q3.txt": {"6d509480ffbf:1001:1", "6d509480ffbf:1002:1", "6d509480ffbf:1003:1"},
		"DEMO24Q3.txt": {"5259d6c52178:1001", "5259d6c52178:1002", "5259d6c52178:1003"},
		"REAC24Q3.txt": {"5e477742087d:1001:myalgia", "5e477742087d:1001:rhabdomyolysis", "5e477742087d:1002:nausea"},
	}
	for name, want := range golden {
		src := testSource(t, name)
		tbl := ParseSource(src, &Warnings{})
		if tbl == nil {
			t.Fatalf("%s: expected table", name)
		}
		srid := indexOf(tbl.Columns, "source_record_id")
		prefix := b3Short(src.SourceB3)
		if !b3PrefixRe.MatchString(prefix) {
			t.Fatalf("%s: b3 prefix %q is not 12 lowercase hex", name, prefix)
		}
		if len(tbl.Rows) != len(want) {
			t.Fatalf("%s: rows = %d, want %d", name, len(tbl.Rows), len(want))
		}
		for i, r := range tbl.Rows {
			got := cell(r, srid)
			if got != want[i] {
				t.Errorf("%s row %d source_record_id = %q, want %q", name, i, got, want[i])
			}
			// Format: <12-hex b3 prefix>:<primaryid>[:<family-specific seq|pt>].
			if !strings.HasPrefix(got, prefix+":") {
				t.Errorf("%s row %d source_record_id %q does not start with b3 prefix %q", name, i, got, prefix)
			}
		}
	}
}

func TestParseSourceRecordIDFamilyStructure(t *testing.T) {
	// Colon-part counts by family: DEMO/RPSR/DELETE = 2, DRUG/REAC = 3, INDI = 4.
	cases := []struct {
		family, content string
		wantParts       int
	}{
		{"DEMO", "PRIMARYID$CASEID$\r\n1$2$\r\n", 2},
		{"RPSR", "PRIMARYID$RPSR_COD$\r\n1$DIRECT$\r\n", 2},
		{"DELETE", "PRIMARYID$CASEID$\r\n1$2$\r\n", 2},
		{"DRUG", "PRIMARYID$DRUG_SEQ$\r\n1$3$\r\n", 3},
		{"REAC", "PRIMARYID$PT$\r\n1$pain$\r\n", 3},
		{"INDI", "PRIMARYID$INDI_DRUG_SEQ$INDI_PT$\r\n1$3$pain$\r\n", 4},
	}
	for _, c := range cases {
		src := inlineSource("24Q3", c.family, c.family+"24Q3.txt", c.content)
		tbl := ParseSource(src, &Warnings{})
		if tbl == nil {
			t.Fatalf("%s: expected table", c.family)
		}
		got := len(strings.Split(cell(tbl.Rows[0], indexOf(tbl.Columns, "source_record_id")), ":"))
		if got != c.wantParts {
			t.Errorf("%s: source_record_id parts = %d, want %d", c.family, got, c.wantParts)
		}
	}
}

func TestNormalizedColumnLayoutProvenanceFirst(t *testing.T) {
	tbl := parseFamilies(t, "DRUG24Q3.txt")["DRUG"]
	want := []string{"quarter", "source_file", "source_record_id", "primaryid", "drug_seq", "drugname", "role_cod", "nda_num", "prod_ai"}
	if !reflectEqual(tbl.Columns, want) {
		t.Fatalf("DRUG columns = %v, want %v", tbl.Columns, want)
	}
	// One stable id per row.
	srid := indexOf(tbl.Columns, "source_record_id")
	seen := map[string]bool{}
	for _, r := range tbl.Rows {
		seen[cell(r, srid)] = true
	}
	if len(seen) != len(tbl.Rows) {
		t.Fatalf("source_record_id not unique per row: %d unique of %d", len(seen), len(tbl.Rows))
	}
}

// --- DELETE filtering ----------------------------------------------------------

func TestDeletedPrimaryIDs(t *testing.T) {
	fam := parseFamilies(t, "DELETE24Q3.txt")
	deleted := DeletedPrimaryIDs(fam["DELETE"])
	if !deleted["1003"] || len(deleted) != 1 {
		t.Fatalf("deleted = %v, want {1003}", deleted)
	}
	if got := DeletedPrimaryIDs(nil); len(got) != 0 {
		t.Fatalf("DeletedPrimaryIDs(nil) = %v, want empty", got)
	}
}

// --- per-quarter case join -----------------------------------------------------

func quarter24Q3Cases(t *testing.T) []Case {
	t.Helper()
	fam := parseFamilies(t, "DEMO24Q3.txt", "DRUG24Q3.txt", "INDI24Q3.txt", "REAC24Q3.txt", "RPSR24Q3.txt", "DELETE24Q3.txt")
	warn := &Warnings{}
	cases := BuildQuarterCases(fam, "24Q3", DeletedPrimaryIDs(fam["DELETE"]), warn)
	return cases
}

func TestBuildQuarterCasesDeleteFilteringAndColumns(t *testing.T) {
	cases := quarter24Q3Cases(t)
	// 1003 is DELETEd -> dropped; 1001 and 1002 survive.
	if len(cases) != 2 {
		t.Fatalf("cases = %d, want 2 (%v)", len(cases), cases)
	}
	byPid := byPrimaryID(cases)
	if _, ok := byPid["1003"]; ok {
		t.Fatal("deleted primaryid 1003 survived the join")
	}
	c := byPid["1001"]
	if c.Quarter != "24Q3" || c.CaseID != "5001" || c.OccpCod != "MD" || c.ReporterCountry != "US" {
		t.Fatalf("1001 reporter metadata wrong: %+v", c)
	}
	if c.Source != "PERIODIC" { // fixture had a trailing space
		t.Fatalf("1001 source = %q, want PERIODIC", c.Source)
	}
	if c.Drugname != "Examplestatin" || c.Ingredient != "Examplestatin" || c.RoleCod != "PS" {
		t.Fatalf("1001 drug fields wrong: %+v", c)
	}
	if c.DrugSeq != "1" || c.IndiDrugSeq != "1" || c.Indication != "hypercholesterolemia" {
		t.Fatalf("1001 seq/indication wrong: %+v", c)
	}
	// drugname (proprietary) differs from ingredient (prod_ai) for the Advil case.
	advil := byPid["1002"]
	if advil.Drugname != "Advil" || advil.Ingredient != "Ibuprofen" {
		t.Fatalf("1002 drugname/ingredient = %q/%q, want Advil/Ibuprofen", advil.Drugname, advil.Ingredient)
	}
}

func TestBuildQuarterCasesNDAAndRaw(t *testing.T) {
	byPid := byPrimaryID(quarter24Q3Cases(t))
	if byPid["1001"].Nda != "12345" || byPid["1001"].NdaRaw != "012345" {
		t.Fatalf("1001 nda/raw = %q/%q, want 12345/012345", byPid["1001"].Nda, byPid["1001"].NdaRaw)
	}
	if byPid["1002"].Nda != "17977" || byPid["1002"].NdaRaw != "017977" {
		t.Fatalf("1002 nda/raw = %q/%q, want 17977/017977", byPid["1002"].Nda, byPid["1002"].NdaRaw)
	}
}

func TestBuildQuarterCasesEffectsSortedUniqueDollarJoined(t *testing.T) {
	byPid := byPrimaryID(quarter24Q3Cases(t))
	if byPid["1001"].Effects != "myalgia$rhabdomyolysis" {
		t.Fatalf("1001 effects = %q, want myalgia$rhabdomyolysis", byPid["1001"].Effects)
	}
	if byPid["1002"].Effects != "nausea" {
		t.Fatalf("1002 effects = %q, want nausea", byPid["1002"].Effects)
	}
}

func TestBuildQuarterCasesSourceRecordIDFormat(t *testing.T) {
	byPid := byPrimaryID(quarter24Q3Cases(t))
	if got := byPid["1001"].SourceRecordID; got != "24Q3:1001:1:hypercholesterolemia" {
		t.Fatalf("1001 case source_record_id = %q, want 24Q3:1001:1:hypercholesterolemia", got)
	}
}

func TestBuildQuarterCasesNoDrugOrIndiYieldsNoCases(t *testing.T) {
	fam := parseFamilies(t, "DEMO24Q3.txt") // DEMO only
	if cases := BuildQuarterCases(fam, "24Q3", nil, &Warnings{}); len(cases) != 0 {
		t.Fatalf("expected no cases without DRUG/INDI, got %d", len(cases))
	}
}

func TestBuildQuarterCasesDeletedRowsWarning(t *testing.T) {
	fam := parseFamilies(t, "DEMO24Q3.txt", "DRUG24Q3.txt", "INDI24Q3.txt", "REAC24Q3.txt", "RPSR24Q3.txt", "DELETE24Q3.txt")
	warn := &Warnings{}
	BuildQuarterCases(fam, "24Q3", DeletedPrimaryIDs(fam["DELETE"]), warn)
	if !warn.Has("deleted_rows_dropped") {
		t.Fatalf("expected deleted_rows_dropped warning, got %v", warn.Items())
	}
}

// --- NDA normalization ---------------------------------------------------------

func TestNormalizeNDA(t *testing.T) {
	cases := []struct{ in, want string }{
		{"012345", "12345"},
		{"020000", "20000"},
		{"000000", ""},     // all leading zeroes -> empty
		{"", ""},           // empty stays empty
		{"NDA-0079", "79"}, // non-digits stripped, then leading zeroes
		{"  0042 ", "42"},  // spaces are non-digits
		{"123", "123"},     // no leading zero
	}
	for _, c := range cases {
		if got := normalizeNDA(c.in); got != c.want {
			t.Errorf("normalizeNDA(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// --- intra-quarter exact-row dedup ---------------------------------------------

func TestIntraQuarterDuplicateIndiRowsDeduped(t *testing.T) {
	fam := map[string]*Table{}
	warn := &Warnings{}
	for _, src := range []Source{
		inlineSource("24Q3", "DEMO", "DEMO24Q3.txt", "PRIMARYID$CASEID$\r\n1001$5001$\r\n"),
		inlineSource("24Q3", "DRUG", "DRUG24Q3.txt", "PRIMARYID$DRUG_SEQ$DRUGNAME$ROLE_COD$NDA_NUM$PROD_AI$\r\n1001$1$DrugX$PS$012345$IngX$\r\n"),
		// Duplicate INDI row for the same (primaryid, drug_seq, pt) must collapse.
		inlineSource("24Q3", "INDI", "INDI24Q3.txt", "PRIMARYID$INDI_DRUG_SEQ$INDI_PT$\r\n1001$1$pain$\r\n1001$1$pain$\r\n"),
	} {
		if tbl := ParseSource(src, warn); tbl != nil {
			fam[src.Family] = tbl
		}
	}
	cases := BuildQuarterCases(fam, "24Q3", nil, warn) // no DELETE -> 1001 not deleted
	if len(cases) != 1 {
		t.Fatalf("cases = %d, want 1 (dup INDI collapsed): %+v", len(cases), cases)
	}
	if cases[0].Indication != "pain" {
		t.Fatalf("indication = %q, want pain", cases[0].Indication)
	}
}

// --- cross-quarter caseid dedup ------------------------------------------------

func TestReduceCasesCaseIDDedupMostRecentWins(t *testing.T) {
	srcs := loadTestdata(t)
	warn := &Warnings{}
	res, err := Extract(context.Background(), srcs, 0, warn)
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	// 1003 deleted; 2001 superseded by 1001 (caseid 5001, 24Q3 wins) -> {1001,1002,2002}.
	if len(res.Cases) != 3 {
		t.Fatalf("cases = %d, want 3: %+v", len(res.Cases), res.Cases)
	}
	byPid := byPrimaryID(res.Cases)
	if _, ok := byPid["2001"]; ok {
		t.Fatal("superseded primaryid 2001 survived cross-quarter dedup")
	}
	for _, pid := range []string{"1001", "1002", "2002"} {
		if _, ok := byPid[pid]; !ok {
			t.Fatalf("expected survivor %q missing", pid)
		}
	}
	if byPid["1001"].Quarter != "24Q3" || byPid["2002"].Quarter != "24Q2" {
		t.Fatalf("winning quarters wrong: 1001=%q 2002=%q", byPid["1001"].Quarter, byPid["2002"].Quarter)
	}
	// Deterministic order: sorted by primaryid.
	if res.Cases[0].PrimaryID != "1001" || res.Cases[1].PrimaryID != "1002" || res.Cases[2].PrimaryID != "2002" {
		t.Fatalf("case order not deterministic: %v", []string{res.Cases[0].PrimaryID, res.Cases[1].PrimaryID, res.Cases[2].PrimaryID})
	}
}

func TestReduceCasesDedupAudit(t *testing.T) {
	res, err := Extract(context.Background(), loadTestdata(t), 0, &Warnings{})
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if len(res.DedupAudit) != 1 {
		t.Fatalf("dedup audit rows = %d, want 1: %+v", len(res.DedupAudit), res.DedupAudit)
	}
	a := res.DedupAudit[0]
	want := DedupAuditRow{Quarter: "24Q2", PrimaryID: "2001", CaseID: "5001", DedupKey: "5001", WinningQuarter: "24Q3", SourceFile: "DRUG24Q2.txt"}
	if a != want {
		t.Fatalf("dedup audit = %+v, want %+v", a, want)
	}
}

func TestReduceCasesDeleteAudit(t *testing.T) {
	res, err := Extract(context.Background(), loadTestdata(t), 0, &Warnings{})
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if len(res.DeleteAudit) != 1 {
		t.Fatalf("delete audit rows = %d, want 1: %+v", len(res.DeleteAudit), res.DeleteAudit)
	}
	a := res.DeleteAudit[0]
	if a.Quarter != "24Q3" || a.PrimaryID != "1003" || a.CaseID != "5004" || a.SourceFile != "DELETE24Q3.txt" {
		t.Fatalf("delete audit = %+v", a)
	}
	if !strings.HasSuffix(a.SourceRecordID, ":1003") {
		t.Fatalf("delete audit source_record_id = %q, want suffix :1003", a.SourceRecordID)
	}
}

func TestSingleQuarterHasNoDedup(t *testing.T) {
	var srcs []Source
	for _, s := range loadTestdata(t) {
		if s.Quarter == "24Q3" {
			srcs = append(srcs, s)
		}
	}
	res, err := Extract(context.Background(), srcs, 0, &Warnings{})
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if len(res.DedupAudit) != 0 {
		t.Fatalf("single quarter should have no dedup audit, got %+v", res.DedupAudit)
	}
}

func TestExtractEmptyInput(t *testing.T) {
	res, err := Extract(context.Background(), nil, 0, &Warnings{})
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if len(res.Cases) != 0 || len(res.DeleteAudit) != 0 || len(res.DedupAudit) != 0 {
		t.Fatalf("empty input should yield empty result, got %+v", res)
	}
}

// --- public TSV contract + parity ----------------------------------------------

func TestCasesTSVColumnsMatchContract(t *testing.T) {
	// Mirrors schemas.FAERS_CASES_COLUMNS exactly.
	want := []string{"quarter", "primaryid", "caseid", "source", "occp_cod", "reporter_country", "drugname", "ingredient", "nda", "indication", "effects"}
	if !reflectEqual(CasesTSVColumns, want) {
		t.Fatalf("CasesTSVColumns = %v, want %v", CasesTSVColumns, want)
	}
}

func TestExtractParityWithPythonTSV(t *testing.T) {
	res, err := Extract(context.Background(), loadTestdata(t), 0, &Warnings{})
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	var buf bytes.Buffer
	if err := WriteCasesTSV(&buf, res.Cases); err != nil {
		t.Fatalf("WriteCasesTSV: %v", err)
	}
	got := buf.String()
	if got != goldenCasesTSV {
		t.Fatalf("TSV parity mismatch\n--- got ---\n%s\n--- want (Python) ---\n%s", got, goldenCasesTSV)
	}
	// Header line is exactly the contract columns; output is uncompressed (no gzip magic).
	header := strings.SplitN(got, "\n", 2)[0]
	if !reflectEqual(strings.Split(header, "\t"), CasesTSVColumns) {
		t.Fatalf("TSV header = %q, want %v", header, CasesTSVColumns)
	}
	if bytes.HasPrefix([]byte(got), []byte{0x1f, 0x8b}) {
		t.Fatal("TSV must be uncompressed, got gzip magic bytes")
	}
}

func TestWriteAuditTSVs(t *testing.T) {
	res, err := Extract(context.Background(), loadTestdata(t), 0, &Warnings{})
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	var del, dedup bytes.Buffer
	if err := WriteDeleteAuditTSV(&del, res.DeleteAudit); err != nil {
		t.Fatal(err)
	}
	if err := WriteDedupAuditTSV(&dedup, res.DedupAudit); err != nil {
		t.Fatal(err)
	}
	if got := strings.SplitN(del.String(), "\n", 2)[0]; !reflectEqual(strings.Split(got, "\t"), DeleteAuditColumns) {
		t.Fatalf("delete audit header = %q", got)
	}
	if got := strings.SplitN(dedup.String(), "\n", 2)[0]; !reflectEqual(strings.Split(got, "\t"), DedupAuditColumns) {
		t.Fatalf("dedup audit header = %q", got)
	}
}

// --- determinism ---------------------------------------------------------------

func TestExtractIsDeterministic(t *testing.T) {
	var prev []byte
	for i := 0; i < 5; i++ {
		res, err := Extract(context.Background(), loadTestdata(t), 3, &Warnings{})
		if err != nil {
			t.Fatalf("Extract run %d: %v", i, err)
		}
		var buf bytes.Buffer
		if err := WriteCasesTSV(&buf, res.Cases); err != nil {
			t.Fatal(err)
		}
		if prev == nil {
			prev = buf.Bytes()
			continue
		}
		if !bytes.Equal(prev, buf.Bytes()) {
			t.Fatalf("run %d produced different TSV bytes", i)
		}
	}
}

// reflectEqual is a tiny dependency-free slice equality helper for tests.
func reflectEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
