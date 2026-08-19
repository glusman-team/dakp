package airflow

import (
	"archive/zip"
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/glusman-team/dakp/go/internal/blake3store"
	"github.com/glusman-team/dakp/go/internal/faers"
)

// streamFixtureRuns runs the production streaming pipeline shape on the committed fixtures:
// per-quarter parse + case join, streaming dedup (most-recent-first), one sorted run file per
// quarter. It returns the run paths (newest-first) and the collected dedup audit.
func streamFixtureRuns(t *testing.T, dir string) ([]string, []faers.DedupAuditRow, int) {
	t.Helper()
	refs := faersFixtureRefs(t)
	srcs, err := loadFAERSSources(mustStage(t, refs))
	if err != nil {
		t.Fatal(err)
	}
	warn := &faers.Warnings{}
	byQuarter := map[string]map[string]*faers.Table{}
	for _, s := range srcs {
		tbl := faers.ParseSource(s, warn)
		if tbl == nil || len(tbl.Rows) == 0 {
			continue
		}
		if byQuarter[s.Quarter] == nil {
			byQuarter[s.Quarter] = map[string]*faers.Table{}
		}
		byQuarter[s.Quarter][s.Family] = tbl
	}
	quarters := make([]string, 0, len(byQuarter))
	for q := range byQuarter {
		quarters = append(quarters, q)
	}
	sort.Slice(quarters, func(i, j int) bool { return quarters[i] > quarters[j] }) // most-recent-first

	deduper := faers.NewDeduper()
	var runs []string
	var audit []faers.DedupAuditRow
	totalKept := 0
	for _, q := range quarters {
		fam := byQuarter[q]
		cases := faers.BuildQuarterCases(fam, q, faers.DeletedPrimaryIDs(fam["DELETE"]), warn)
		kept, superseded, newKeys := deduper.Split(q, cases)
		deduper.Commit(q, newKeys)
		audit = append(audit, superseded...)
		run := filepath.Join(dir, "run_"+q+".parquet")
		if err := writeFAERSRun(run, kept); err != nil {
			t.Fatalf("write run %s: %v", q, err)
		}
		runs = append(runs, run)
		totalKept += len(kept)
	}
	return runs, audit, totalKept
}

// TestStreamingPipelineMatchesBatch is THE parity gate for the bounded-memory rewrite: the
// streaming pipeline (per-quarter build -> dedup -> sorted runs -> k-way merge) must emit the
// exact faers_cases.tsv bytes of the legacy batch path (faers.Extract + WriteCasesTSV) on the
// committed fixtures — same rows, same global (primaryid, drug_seq, indication) order with
// newer-quarter tiebreak.
func TestStreamingPipelineMatchesBatch(t *testing.T) {
	refs := faersFixtureRefs(t)
	srcs, err := loadFAERSSources(mustStage(t, refs))
	if err != nil {
		t.Fatal(err)
	}
	res, err := faers.Extract(context.Background(), srcs, 4, &faers.Warnings{})
	if err != nil {
		t.Fatal(err)
	}
	var wantTSV bytes.Buffer
	if err := faers.WriteCasesTSV(&wantTSV, res.Cases); err != nil {
		t.Fatal(err)
	}

	runs, audit, totalKept := streamFixtureRuns(t, t.TempDir())

	// The streaming dedup must keep exactly the batch survivors and produce the same audit.
	if totalKept != len(res.Cases) {
		t.Fatalf("streaming kept %d cases, batch kept %d", totalKept, len(res.Cases))
	}
	faers.SortDedupAudit(audit)
	if len(audit) != len(res.DedupAudit) {
		t.Fatalf("streaming dedup audit = %d rows, want %d", len(audit), len(res.DedupAudit))
	}
	for i := range audit {
		if audit[i] != res.DedupAudit[i] {
			t.Fatalf("audit[%d] = %+v, want %+v", i, audit[i], res.DedupAudit[i])
		}
	}

	// Merged TSV bytes must match the batch TSV bytes exactly.
	var gotTSV bytes.Buffer
	if _, err := gotTSV.WriteString(joinTabs(faers.CasesTSVColumns) + "\n"); err != nil {
		t.Fatal(err)
	}
	merged := 0
	err = mergeFAERSRuns(runs, func(row []string) error {
		merged++
		return writeFAERSRunTSVRow(&gotTSV, row)
	})
	if err != nil {
		t.Fatalf("merge: %v", err)
	}
	if merged != len(res.Cases) {
		t.Fatalf("merged %d rows, want %d", merged, len(res.Cases))
	}
	if !bytes.Equal(gotTSV.Bytes(), wantTSV.Bytes()) {
		t.Errorf("streaming TSV != batch TSV\n--- got ---\n%s\n--- want ---\n%s", gotTSV.String(), wantTSV.String())
	}
}

// TestMergeFAERSRunsEmptyAndMissing covers degenerate runs: an empty run file is skipped and a
// missing run file errors.
func TestMergeFAERSRunsEmptyAndMissing(t *testing.T) {
	dir := t.TempDir()
	empty := filepath.Join(dir, "empty.parquet")
	if err := writeFAERSRun(empty, nil); err != nil {
		t.Fatal(err)
	}
	rows := 0
	if err := mergeFAERSRuns([]string{empty}, func([]string) error { rows++; return nil }); err != nil {
		t.Fatalf("merge(empty): %v", err)
	}
	if rows != 0 {
		t.Fatalf("merge(empty) emitted %d rows, want 0", rows)
	}
	if err := mergeFAERSRuns([]string{filepath.Join(dir, "missing.parquet")}, func([]string) error { return nil }); err == nil {
		t.Fatal("merge(missing) should error")
	}
}

// TestFAERSRunRoundTrip checks the run writer/reader round-trips every field in contract order.
func TestFAERSRunRoundTrip(t *testing.T) {
	cases := []faers.Case{
		{
			Quarter: "24Q3", PrimaryID: "1001", CaseID: "5001", Source: "PERIODIC", OccpCod: "MD",
			ReporterCountry: "US", Drugname: "Examplestatin", Ingredient: "Examplestatin", Nda: "12345",
			Indication: "hypercholesterolemia", Effects: "myalgia$rhabdomyolysis", DrugSeq: "1",
		},
		{
			Quarter: "24Q3", PrimaryID: "1002", CaseID: "5002", Source: "DIRECT", OccpCod: "PH",
			ReporterCountry: "GB", Drugname: "Advil", Ingredient: "Ibuprofen", Nda: "17977",
			Indication: "headache", Effects: "nausea", DrugSeq: "1",
		},
	}
	path := filepath.Join(t.TempDir(), "run.parquet")
	if err := writeFAERSRun(path, cases); err != nil {
		t.Fatal(err)
	}
	rd, err := openFAERSRun(path)
	if err != nil {
		t.Fatal(err)
	}
	defer rd.Close()
	var got [][]string
	for {
		ok, err := rd.next()
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
		got = append(got, append([]string{}, rd.cur...))
	}
	if len(got) != len(cases) {
		t.Fatalf("round-trip rows = %d, want %d", len(got), len(cases))
	}
	for i, c := range cases {
		want := map[string]string{
			"quarter": c.Quarter, "primaryid": c.PrimaryID, "caseid": c.CaseID, "source": c.Source,
			"occp_cod": c.OccpCod, "reporter_country": c.ReporterCountry, "drugname": c.Drugname,
			"ingredient": c.Ingredient, "nda": c.Nda, "nda_raw": c.NdaRaw, "role_cod": c.RoleCod,
			"drug_seq": c.DrugSeq, "indi_drug_seq": c.IndiDrugSeq, "indication": c.Indication,
			"effects": c.Effects, "source_file": c.SourceFile, "source_record_id": c.SourceRecordID,
		}
		for name, expected := range want {
			if gotValue := got[i][faersRunColumnIndex[name]]; gotValue != expected {
				t.Errorf("row %d col %s = %q, want %q", i, name, gotValue, expected)
			}
		}
	}
}

// TestStreamingCasesParquetMatchesBatch verifies the merged cases.parquet emission (the full rich
// case row, including provenance columns) row-for-row against the batch path.
func TestStreamingCasesParquetMatchesBatch(t *testing.T) {
	refs := faersFixtureRefs(t)
	srcs, err := loadFAERSSources(mustStage(t, refs))
	if err != nil {
		t.Fatal(err)
	}
	res, err := faers.Extract(context.Background(), srcs, 4, &faers.Warnings{})
	if err != nil {
		t.Fatal(err)
	}
	runs, _, _ := streamFixtureRuns(t, t.TempDir())

	path := filepath.Join(t.TempDir(), "cases.parquet")
	w, err := NewStringParquetWriter(path, faersCaseColumns)
	if err != nil {
		t.Fatal(err)
	}
	err = mergeFAERSRuns(runs, func(row []string) error { return w.Append(faersCaseRow17(row)) })
	if err != nil {
		t.Fatal(err)
	}
	n, err := w.Close()
	if err != nil {
		t.Fatal(err)
	}
	if n != len(res.Cases) {
		t.Fatalf("cases.parquet rows = %d, want %d", n, len(res.Cases))
	}
	cols, rows := readBack(t, path)
	if len(cols) != len(faersCaseColumns) {
		t.Fatalf("cases.parquet columns = %v, want %d", cols, len(faersCaseColumns))
	}
	colIdx := make(map[string]int, len(cols))
	for i, c := range cols {
		colIdx[c] = i
	}
	for i, c := range res.Cases {
		want := map[string]string{
			"quarter": c.Quarter, "primaryid": c.PrimaryID, "caseid": c.CaseID, "source": c.Source,
			"occp_cod": c.OccpCod, "reporter_country": c.ReporterCountry, "drugname": c.Drugname,
			"ingredient": c.Ingredient, "nda": c.Nda, "indication": c.Indication, "effects": c.Effects,
			"nda_raw": c.NdaRaw, "role_cod": c.RoleCod, "drug_seq": c.DrugSeq, "indi_drug_seq": c.IndiDrugSeq,
			"source_file": c.SourceFile, "source_record_id": c.SourceRecordID,
		}
		for name, w := range want {
			if got := rows[i][colIdx[name]]; got != w {
				t.Fatalf("row %d col %s = %q, want %q", i, name, got, w)
			}
		}
	}
}

// TestExtractFAERSZipStreamingParity feeds the streaming ExtractFAERS the real download shape
// (quarterly ASCII zips with members nested under ASCII/) and requires the exact same
// faers_cases.tsv bytes as the loose-fixture run — locking the lazy zip inventory + streaming
// parse + merge path end to end.
func TestExtractFAERSZipStreamingParity(t *testing.T) {
	looseRefs := faersFixtureRefs(t)
	looseOut, err := ExtractFAERS(context.Background(), Config{Workdir: t.TempDir(), Profile: "mock", Threads: 4}, looseRefs)
	if err != nil {
		t.Fatalf("ExtractFAERS(loose): %v", err)
	}
	looseTSV, err := os.ReadFile(looseOut[1].URI)
	if err != nil {
		t.Fatal(err)
	}

	// Pack the fixtures into one zip per quarter (members nested under ASCII/, the real shape).
	zipDir := t.TempDir()
	var zipRefs []ArtifactRef
	for _, q := range []string{"24Q3", "24Q2"} {
		zipPath := filepath.Join(zipDir, fmt.Sprintf("faers_ascii_%s.zip", q))
		f, err := os.Create(zipPath)
		if err != nil {
			t.Fatal(err)
		}
		zw := zip.NewWriter(f)
		for _, src := range looseRefs {
			name := filepath.Base(src.URI)
			if _, quarter := faers.FamilyAndQuarter(name); quarter != q {
				continue
			}
			body, err := os.ReadFile(src.URI)
			if err != nil {
				t.Fatal(err)
			}
			w, err := zw.Create("ASCII/" + name)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := w.Write(body); err != nil {
				t.Fatal(err)
			}
		}
		// A non-FAERS member must be skipped without breaking the run.
		if w, err := zw.Create("ASCII/Readme.pdf"); err != nil {
			t.Fatal(err)
		} else if _, err := w.Write([]byte("not ascii")); err != nil {
			t.Fatal(err)
		}
		if err := zw.Close(); err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		id, err := blake3store.HashFile(zipPath)
		if err != nil {
			t.Fatal(err)
		}
		zipRefs = append(zipRefs, ArtifactRef{URI: zipPath, Blake3: id, MediaType: "application/zip"})
	}

	zipOut, err := ExtractFAERS(context.Background(), Config{Workdir: t.TempDir(), Profile: "mock", Threads: 4}, zipRefs)
	if err != nil {
		t.Fatalf("ExtractFAERS(zip): %v", err)
	}
	zipTSV, err := os.ReadFile(zipOut[1].URI)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(zipTSV, looseTSV) {
		t.Errorf("zip TSV != loose TSV\n--- zip ---\n%s\n--- loose ---\n%s", zipTSV, looseTSV)
	}
	if *zipOut[0].Rows != *looseOut[0].Rows {
		t.Errorf("zip cases rows = %d, want %d", *zipOut[0].Rows, *looseOut[0].Rows)
	}
}

// joinTabs joins strings with tabs (test helper keeping the TSV header literal).
func joinTabs(parts []string) string {
	var b bytes.Buffer
	for i, p := range parts {
		if i > 0 {
			b.WriteByte('\t')
		}
		b.WriteString(p)
	}
	return b.String()
}
