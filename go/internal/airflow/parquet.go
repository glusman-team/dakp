// Package airflow holds the support layer for DAKP's native Airflow Go SDK bundle workers:
// the ArtifactRef/Config XCom (de)serialization, input staging, BLAKE3 store registration
// (mirroring Python io/artifact_store.py), and a generic all-string parquet writer for the
// interim tables. It is the Go counterpart to the Python io/ contracts, kept parity-compatible
// so the unchanged Python shaping stage can consume the Go-produced interim artifacts.
package airflow

import (
	"os"

	"github.com/parquet-go/parquet-go"
)

// parquetMaxRowsPerRowGroup caps rows per parquet row group so writers stay bounded in memory:
// the parquet-go default is math.MaxInt64 (one row group == the whole table buffered). DAKP's
// small normalized tables still fit a single row group (byte-identical output); FAERS-scale
// outputs get many row groups instead of one giant in-memory one.
const parquetMaxRowsPerRowGroup = 1 << 20

// WriteStringParquet writes columns + rows as an all-string (UTF8) parquet file and returns the
// row count. Every DAKP interim table is all-Utf8 (the Python extractor builds all-string frames
// and the Go TSV parity reads with infer_schema_length=0), so a single generic string writer
// covers all of them. Column *names* match the Python contracts exactly. parquet.Group is a map,
// so parquet-go writes the columns in alphabetical physical order; this is immaterial because
// polars reads parquet by column name (the shaping stage selects columns by name) and the
// schema_fingerprint is computed from the logical contract order, not the physical order
// (verified: polars reads Go-written parquet with the correct String schema and exact values).
// Empty cells are written as "" (never null), matching the Python all-string frames.
// Streaming callers (FAERS-scale outputs) use NewStringParquetWriter directly.
func WriteStringParquet(path string, columns []string, rows [][]string) (int, error) {
	w, err := NewStringParquetWriter(path, columns)
	if err != nil {
		return 0, err
	}
	for _, row := range rows {
		if err := w.Append(row); err != nil {
			w.Close()
			return 0, err
		}
	}
	return w.Close()
}

// StringParquetWriter streams all-string (UTF8) rows into one parquet file with bounded memory
// (row groups capped at parquetMaxRowsPerRowGroup rows). Same schema/leaf mapping as
// WriteStringParquet: rows smaller than the cap produce byte-identical files.
type StringParquetWriter struct {
	f       *os.File
	w       *parquet.Writer
	columns []string
	leafIdx map[string]int
	rows    int
}

// NewStringParquetWriter opens path for streaming all-string parquet rows under the given
// (logical contract) column names.
func NewStringParquetWriter(path string, columns []string) (*StringParquetWriter, error) {
	group := make(parquet.Group, len(columns))
	for _, col := range columns {
		group[col] = parquet.String()
	}
	schema := parquet.NewSchema("table", group)

	// Map column name -> leaf index in the schema's (deterministic) leaf order, so each value can
	// be placed at the correct column index regardless of the group map's iteration order.
	leafIdx := make(map[string]int, len(columns))
	for i, colPath := range schema.Columns() {
		leafIdx[colPath[len(colPath)-1]] = i
	}

	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	w := parquet.NewWriter(f, schema, parquet.MaxRowsPerRowGroup(parquetMaxRowsPerRowGroup))
	return &StringParquetWriter{f: f, w: w, columns: columns, leafIdx: leafIdx}, nil
}

// Append writes one row (values aligned to the writer's columns; short rows pad with "").
func (s *StringParquetWriter) Append(row []string) error {
	r := make(parquet.Row, len(s.columns))
	for ci, col := range s.columns {
		v := ""
		if ci < len(row) {
			v = row[ci]
		}
		idx := s.leafIdx[col]
		r[idx] = parquet.ValueOf(v).Level(0, 0, idx) // flat required column: rep=0, def=0
	}
	if _, err := s.w.WriteRows([]parquet.Row{r}); err != nil {
		return err
	}
	s.rows++
	return nil
}

// Close finalizes the parquet file and returns the number of rows written.
func (s *StringParquetWriter) Close() (int, error) {
	if err := s.w.Close(); err != nil {
		s.f.Close()
		return s.rows, err
	}
	return s.rows, s.f.Close()
}
