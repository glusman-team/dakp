package airflow

import (
	"io"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/parquet-go/parquet-go"
)

// readBack opens a parquet file and returns its column names (schema leaf order) and rows
// (each row's values in column-index order, decoded as strings).
func readBack(t *testing.T, path string) ([]string, [][]string) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	pf, err := parquet.OpenFile(f, info.Size())
	if err != nil {
		t.Fatal(err)
	}
	var cols []string
	for _, colPath := range pf.Schema().Columns() {
		cols = append(cols, colPath[len(colPath)-1])
	}
	// column name -> index, to reorder each row's values into `cols` order
	idx := make(map[string]int, len(cols))
	for i, c := range cols {
		idx[c] = i
	}

	var rows [][]string
	for _, rg := range pf.RowGroups() {
		r := rg.Rows()
		buf := make([]parquet.Row, 64)
		for {
			n, err := r.ReadRows(buf)
			for i := 0; i < n; i++ {
				rec := make([]string, len(cols))
				for _, v := range buf[i] {
					rec[idx[cols[v.Column()]]] = v.String()
				}
				rows = append(rows, rec)
			}
			if err == io.EOF {
				break
			}
			if err != nil {
				r.Close()
				t.Fatal(err)
			}
		}
		r.Close()
	}
	return cols, rows
}

func TestWriteStringParquetRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "t.parquet")
	columns := []string{"spl_set_id", "approval_code", "section_text"}
	rows := [][]string{
		{"set-1", "NDA123", "indications text"},
		{"set-2", "", "tab\tand \"quotes\" and empty next"},
		{"set-3", "BLA9", ""},
	}
	n, err := WriteStringParquet(path, columns, rows)
	if err != nil {
		t.Fatalf("write: %v", err)
	}
	if n != len(rows) {
		t.Fatalf("row count = %d, want %d", n, len(rows))
	}

	gotCols, gotRows := readBack(t, path)
	// Column names must match exactly (as a set; physical order is irrelevant to polars).
	wantSet := append([]string(nil), columns...)
	gotSet := append([]string(nil), gotCols...)
	sort.Strings(wantSet)
	sort.Strings(gotSet)
	if len(gotSet) != len(wantSet) {
		t.Fatalf("columns = %v, want %v", gotCols, columns)
	}
	for i := range wantSet {
		if gotSet[i] != wantSet[i] {
			t.Fatalf("columns = %v, want %v", gotCols, columns)
		}
	}
	// Row values must round-trip exactly (including empty strings and embedded tab/quotes).
	if len(gotRows) != len(rows) {
		t.Fatalf("read %d rows, want %d: %v", len(gotRows), len(rows), gotRows)
	}
	// Reorder got rows into the original `columns` order for comparison.
	colIdx := make(map[string]int, len(gotCols))
	for i, c := range gotCols {
		colIdx[c] = i
	}
	for ri, want := range rows {
		for ci, col := range columns {
			if got := gotRows[ri][colIdx[col]]; got != want[ci] {
				t.Errorf("row %d col %q = %q, want %q", ri, col, got, want[ci])
			}
		}
	}
}
