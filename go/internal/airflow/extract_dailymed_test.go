package airflow

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/glusman-team/dakp/go/internal/blake3store"
	"github.com/glusman-team/dakp/go/internal/dailymed"
)

// dailymedFixture is byte-identical to the Python test fixture; the internal/dailymed
// TestGoldenTSVParity proves its parsed rows equal the Python/polars goldens, so parity here is
// established by composition (parser rows == golden; parquet writer preserves rows exactly).
const dailymedFixtureRel = "../dailymed/testdata/dailymed_spl.xml.gz"

func TestExtractDailyMedParity(t *testing.T) {
	fixture, err := filepath.Abs(dailymedFixtureRel)
	if err != nil {
		t.Fatal(err)
	}
	inputID, err := blake3store.HashFile(fixture)
	if err != nil {
		t.Fatal(err)
	}
	input := ArtifactRef{URI: fixture, Blake3: inputID, MediaType: "application/gzip"}

	cfg := Config{Workdir: t.TempDir(), Profile: "mock", Threads: 4}
	refs, err := ExtractDailyMed(context.Background(), cfg, []ArtifactRef{input})
	if err != nil {
		t.Fatalf("ExtractDailyMed: %v", err)
	}

	// Six refs in the locked order: 5 interim parquet (spl_documents first) + sections TSV handoff.
	wantNames := []string{
		"spl_documents.parquet",
		"spl_sets.parquet",
		"spl_approvals.parquet",
		"spl_ingredients.parquet",
		"spl_sections.parquet",
		"dailymed_spl_sections.tsv",
	}
	if len(refs) != len(wantNames) {
		t.Fatalf("got %d refs, want %d", len(refs), len(wantNames))
	}
	for i, want := range wantNames {
		if got := filepath.Base(refs[i].URI); got != want {
			t.Errorf("ref[%d] basename = %q, want %q", i, got, want)
		}
		if refs[i].Blake3 == "" || refs[i].Blake3[:3] != "b3:" {
			t.Errorf("ref[%d] blake3 = %q, want b3:<hex>", i, refs[i].Blake3)
		}
		if refs[i].Manifest == nil || !fileExists(t, *refs[i].Manifest) {
			t.Errorf("ref[%d] manifest missing", i)
		}
	}
	// Media types: first five parquet, last TSV.
	for i := 0; i < 5; i++ {
		if refs[i].MediaType != ParquetMediaType {
			t.Errorf("ref[%d] media_type = %q, want parquet", i, refs[i].MediaType)
		}
	}
	if refs[5].MediaType != TSVMediaType {
		t.Errorf("ref[5] media_type = %q, want tsv", refs[5].MediaType)
	}

	// The sections TSV handoff must be byte-identical to the Python/polars golden.
	gotTSV, err := os.ReadFile(refs[5].URI)
	if err != nil {
		t.Fatal(err)
	}
	wantTSV, err := os.ReadFile("../dailymed/testdata/golden/spl_sections.tsv")
	if err != nil {
		t.Fatal(err)
	}
	if string(gotTSV) != string(wantTSV) {
		t.Errorf("sections TSV != Python/polars golden\n--- got ---\n%s\n--- want ---\n%s", gotTSV, wantTSV)
	}

	// Each interim parquet reads back with the contract columns (as a set) and the exact row count
	// the parser produced (proving no rows were dropped on the way to parquet).
	tables, err := dailymed.Extract(context.Background(), []string{fixture}, nil, 4)
	if err != nil {
		t.Fatal(err)
	}
	wantRows := map[string]int{
		"spl_documents.parquet":   len(tables.Documents),
		"spl_sets.parquet":        len(tables.Sets),
		"spl_approvals.parquet":   len(tables.Approvals),
		"spl_ingredients.parquet": len(tables.Ingredients),
		"spl_sections.parquet":    len(tables.Sections),
	}
	wantCols := map[string][]string{
		"spl_documents.parquet":   dailymed.DocumentsColumns,
		"spl_sets.parquet":        dailymed.SetsColumns,
		"spl_approvals.parquet":   dailymed.ApprovalsColumns,
		"spl_ingredients.parquet": dailymed.IngredientsColumns,
		"spl_sections.parquet":    dailymed.SectionsColumns,
	}
	for i := 0; i < 5; i++ {
		name := filepath.Base(refs[i].URI)
		cols, rows := readBack(t, refs[i].URI)
		if len(rows) != wantRows[name] {
			t.Errorf("%s: parquet rows = %d, want %d", name, len(rows), wantRows[name])
		}
		gotSet := append([]string(nil), cols...)
		wantSet := append([]string(nil), wantCols[name]...)
		sort.Strings(gotSet)
		sort.Strings(wantSet)
		if len(gotSet) != len(wantSet) {
			t.Errorf("%s: columns = %v, want %v", name, cols, wantCols[name])
			continue
		}
		for j := range wantSet {
			if gotSet[j] != wantSet[j] {
				t.Errorf("%s: columns = %v, want %v", name, cols, wantCols[name])
				break
			}
		}
		if refs[i].Rows == nil || int(*refs[i].Rows) != wantRows[name] {
			t.Errorf("%s: ref.Rows = %v, want %d", name, refs[i].Rows, wantRows[name])
		}
	}
}

func fileExists(t *testing.T, path string) bool {
	t.Helper()
	_, err := os.Stat(path)
	return err == nil
}
