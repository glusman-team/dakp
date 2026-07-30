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

// WriteStringParquet writes columns + rows as an all-string (UTF8) parquet file and returns the
// row count. Every DAKP interim table is all-Utf8 (the Python extractor builds all-string frames
// and the Go TSV parity reads with infer_schema_length=0), so a single generic string writer
// covers all of them. Column *names* match the Python contracts exactly. parquet.Group is a map,
// so parquet-go writes the columns in alphabetical physical order; this is immaterial because
// polars reads parquet by column name (the shaping stage selects columns by name) and the
// schema_fingerprint is computed from the logical contract order, not the physical order
// (verified: polars reads Go-written parquet with the correct String schema and exact values).
// Empty cells are written as "" (never null), matching the Python all-string frames.
func WriteStringParquet(path string, columns []string, rows [][]string) (int, error) {
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
		return 0, err
	}
	w := parquet.NewWriter(f, schema) // *Schema satisfies WriterOption via ConfigureWriter

	for _, row := range rows {
		r := make(parquet.Row, len(columns))
		for ci, col := range columns {
			v := ""
			if ci < len(row) {
				v = row[ci]
			}
			idx := leafIdx[col]
			r[idx] = parquet.ValueOf(v).Level(0, 0, idx) // flat required column: rep=0, def=0
		}
		if _, err := w.WriteRows([]parquet.Row{r}); err != nil {
			f.Close()
			return 0, err
		}
	}
	if err := w.Close(); err != nil {
		f.Close()
		return 0, err
	}
	return len(rows), f.Close()
}
