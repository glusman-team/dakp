package airflow

import (
	"archive/zip"
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/glusman-team/dakp/go/internal/blake3store"
	"github.com/glusman-team/dakp/go/internal/faers"
)

// faersFixtureRefs builds an ArtifactRef per .txt fixture under ../faers/testdata (byte-identical
// to the Python fixtures; internal/faers parity-locks the parser to the Python reference).
func faersFixtureRefs(t *testing.T) []ArtifactRef {
	t.Helper()
	files, err := filepath.Glob("../faers/testdata/*.txt")
	if err != nil || len(files) == 0 {
		t.Fatalf("faers fixtures: %v (%d)", err, len(files))
	}
	var refs []ArtifactRef
	for _, f := range files {
		abs, _ := filepath.Abs(f)
		id, err := blake3store.HashFile(abs)
		if err != nil {
			t.Fatal(err)
		}
		refs = append(refs, ArtifactRef{URI: abs, Blake3: id, MediaType: "text/plain"})
	}
	return refs
}

func TestExtractFAERSParity(t *testing.T) {
	refs := faersFixtureRefs(t)
	cfg := Config{Workdir: t.TempDir(), Profile: "mock", Threads: 4}
	got, err := ExtractFAERS(context.Background(), cfg, refs)
	if err != nil {
		t.Fatalf("ExtractFAERS: %v", err)
	}

	wantNames := []string{"cases.parquet", "faers_cases.tsv", "delete_audit.parquet", "dedup_audit.parquet", "warnings.parquet"}
	if len(got) != len(wantNames) {
		t.Fatalf("got %d refs, want %d", len(got), len(wantNames))
	}
	for i, want := range wantNames {
		if filepath.Base(got[i].URI) != want {
			t.Errorf("ref[%d] = %q, want %q", i, filepath.Base(got[i].URI), want)
		}
	}

	// Canonical parser output for cross-checks.
	srcs, err := loadFAERSSources(mustStage(t, refs))
	if err != nil {
		t.Fatal(err)
	}
	res, err := faers.Extract(context.Background(), srcs, 4, &faers.Warnings{})
	if err != nil {
		t.Fatal(err)
	}

	// faers_cases.tsv must byte-match the canonical parser TSV.
	var wantTSV bytes.Buffer
	if err := faers.WriteCasesTSV(&wantTSV, res.Cases); err != nil {
		t.Fatal(err)
	}
	gotTSV, err := os.ReadFile(got[1].URI)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(gotTSV, wantTSV.Bytes()) {
		t.Errorf("faers_cases.tsv != canonical parser output\n--- got ---\n%s\n--- want ---\n%s", gotTSV, wantTSV.Bytes())
	}

	// cases.parquet: 17 columns (faersCaseColumns) + one row per case.
	cols, rows := readBack(t, got[0].URI)
	if len(cols) != len(faersCaseColumns) {
		t.Errorf("cases.parquet columns = %d (%v), want %d", len(cols), cols, len(faersCaseColumns))
	}
	if len(rows) != len(res.Cases) {
		t.Errorf("cases.parquet rows = %d, want %d", len(rows), len(res.Cases))
	}
	if got[0].Rows == nil || int(*got[0].Rows) != len(res.Cases) {
		t.Errorf("cases.parquet ref.Rows = %v, want %d", got[0].Rows, len(res.Cases))
	}
	colIndex := make(map[string]int, len(cols))
	for i, name := range cols {
		colIndex[name] = i
	}
	for i, want := range res.Cases {
		row := rows[i]
		checks := map[string]string{
			"nda_raw":          want.NdaRaw,
			"drug_seq":         want.DrugSeq,
			"indi_drug_seq":    want.IndiDrugSeq,
			"source_file":      want.SourceFile,
			"source_record_id": want.SourceRecordID,
		}
		for name, expected := range checks {
			if gotValue := row[colIndex[name]]; gotValue != expected {
				t.Errorf("cases.parquet row %d %s = %q, want %q", i, name, gotValue, expected)
			}
		}
	}

	// warnings.parquet is empty (0 rows).
	_, wrows := readBack(t, got[4].URI)
	if len(wrows) != 0 {
		t.Errorf("warnings.parquet rows = %d, want 0", len(wrows))
	}
}

// mustStage stages refs into a fresh dir and returns it (for loading canonical sources).
func mustStage(t *testing.T, refs []ArtifactRef) string {
	t.Helper()
	dir := t.TempDir()
	if err := StageInputs(refs, dir); err != nil {
		t.Fatal(err)
	}
	return dir
}

// TestLoadFAERSSourcesFromZip covers the real download shape: a quarterly ASCII zip nests the .txt
// files under ASCII/ (plus non-ASCII members to skip); loadFAERSSources must read the .txt members
// straight out of the archive (mirrors faers_ascii._iter_faers_sources).
func TestLoadFAERSSourcesFromZip(t *testing.T) {
	dir := t.TempDir()
	zipPath := filepath.Join(dir, "faers_ascii_24Q3.zip")
	f, err := os.Create(zipPath)
	if err != nil {
		t.Fatal(err)
	}
	zw := zip.NewWriter(f)
	for _, name := range []string{"DEMO24Q3.txt", "DRUG24Q3.txt"} {
		body, err := os.ReadFile(filepath.Join("../faers/testdata", name))
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
	// A non-FAERS member (no family prefix) must be skipped.
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
	srcs, err := loadFAERSSources(mustStage(t, []ArtifactRef{{URI: zipPath, Blake3: id, MediaType: "application/zip"}}))
	if err != nil {
		t.Fatal(err)
	}
	if len(srcs) != 2 {
		t.Fatalf("got %d sources, want 2 (DEMO + DRUG from the zip)", len(srcs))
	}
	byFamily := map[string]faers.Source{}
	for _, s := range srcs {
		byFamily[s.Family] = s
	}
	for _, fam := range []string{"DEMO", "DRUG"} {
		s, ok := byFamily[fam]
		if !ok {
			t.Fatalf("missing %s source", fam)
		}
		if s.Quarter != "24Q3" {
			t.Errorf("%s quarter = %q, want 24Q3", fam, s.Quarter)
		}
		want, _ := os.ReadFile(filepath.Join("../faers/testdata", fam+"24Q3.txt"))
		if !bytes.Equal(s.Content, want) {
			t.Errorf("%s content mismatch", fam)
		}
	}
}
