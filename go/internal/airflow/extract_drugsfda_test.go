package airflow

import (
	"archive/zip"
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/glusman-team/dakp/go/internal/blake3store"
	"github.com/glusman-team/dakp/go/internal/drugsfda"
)

func drugsfdaFixtureRefs(t *testing.T) []ArtifactRef {
	t.Helper()
	files, err := filepath.Glob("../drugsfda/testdata/*.tsv")
	if err != nil || len(files) == 0 {
		t.Fatalf("drugsfda fixtures: %v (%d)", err, len(files))
	}
	var refs []ArtifactRef
	for _, f := range files {
		abs, _ := filepath.Abs(f)
		id, err := blake3store.HashFile(abs)
		if err != nil {
			t.Fatal(err)
		}
		refs = append(refs, ArtifactRef{URI: abs, Blake3: id, MediaType: "text/tab-separated-values"})
	}
	return refs
}

func TestExtractDrugsFDAParity(t *testing.T) {
	refs := drugsfdaFixtureRefs(t)
	cfg := Config{Workdir: t.TempDir(), Profile: "mock", Threads: 4}
	got, err := ExtractDrugsFDA(context.Background(), cfg, refs)
	if err != nil {
		t.Fatalf("ExtractDrugsFDA: %v", err)
	}

	// products.parquet, drugsfda_products.tsv, applications.parquet, submissions.parquet,
	// lookups.parquet, extract_warnings.jsonl
	wantNames := []string{
		"products.parquet", "drugsfda_products.tsv", "applications.parquet",
		"submissions.parquet", "lookups.parquet", "extract_warnings.jsonl",
	}
	if len(got) != len(wantNames) {
		t.Fatalf("got %d refs (%v), want %d", len(got), refNames(got), len(wantNames))
	}
	for i, want := range wantNames {
		if filepath.Base(got[i].URI) != want {
			t.Errorf("ref[%d] = %q, want %q", i, filepath.Base(got[i].URI), want)
		}
	}

	// Canonical parser output for row-count cross-checks.
	discovered, err := discoverDrugsFDAInputs(mustStage(t, refs))
	if err != nil {
		t.Fatal(err)
	}
	tables, err := parseDrugsFDAInputs(discovered)
	if err != nil {
		t.Fatal(err)
	}
	res := drugsfda.Extract(tables)

	// Public products TSV must byte-match the Python/polars golden.
	gotTSV, err := os.ReadFile(got[1].URI)
	if err != nil {
		t.Fatal(err)
	}
	wantTSV, err := os.ReadFile("../drugsfda/testdata/golden/drugsfda_products.tsv")
	if err != nil {
		t.Fatal(err)
	}
	if string(gotTSV) != string(wantTSV) {
		t.Errorf("drugsfda_products.tsv != golden\n--- got ---\n%s\n--- want ---\n%s", gotTSV, wantTSV)
	}

	// products.parquet: ProductsColumns + one row per product.
	cols, rows := readBack(t, got[0].URI)
	if len(cols) != len(drugsfda.ProductsColumns) {
		t.Errorf("products.parquet columns = %d, want %d", len(cols), len(drugsfda.ProductsColumns))
	}
	if len(rows) != len(res.Products) {
		t.Errorf("products.parquet rows = %d, want %d", len(rows), len(res.Products))
	}
}

func refNames(refs []ArtifactRef) []string {
	var out []string
	for _, r := range refs {
		out = append(out, filepath.Base(r.URI))
	}
	return out
}

// TestExtractDrugsFDAFromZip covers the real download shape: the Drugs@FDA data-files zip carries
// Products/Applications/Submissions as members at the archive root. StageInputs only links the zip
// into inDir, so ExtractDrugsFDA must unpack the members and extract them like the loose fixtures.
func TestExtractDrugsFDAFromZip(t *testing.T) {
	files, err := filepath.Glob("../drugsfda/testdata/*.tsv")
	if err != nil || len(files) == 0 {
		t.Fatalf("drugsfda fixtures: %v (%d)", err, len(files))
	}
	dir := t.TempDir()
	zipPath := filepath.Join(dir, "drugsfda_data_files.zip")
	f, err := os.Create(zipPath)
	if err != nil {
		t.Fatal(err)
	}
	zw := zip.NewWriter(f)
	for _, fpath := range files {
		body, err := os.ReadFile(fpath)
		if err != nil {
			t.Fatal(err)
		}
		w, err := zw.Create(filepath.Base(fpath))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write(body); err != nil {
			t.Fatal(err)
		}
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
	cfg := Config{Workdir: t.TempDir(), Profile: "mock", Threads: 4}
	got, err := ExtractDrugsFDA(context.Background(), cfg, []ArtifactRef{{URI: zipPath, Blake3: id, MediaType: "application/zip"}})
	if err != nil {
		t.Fatalf("ExtractDrugsFDA(zip): %v", err)
	}
	if len(got) == 0 || filepath.Base(got[0].URI) != "products.parquet" {
		t.Fatalf("got %v, want products.parquet first", refNames(got))
	}
}
