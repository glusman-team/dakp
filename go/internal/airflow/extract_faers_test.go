package airflow

import (
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
