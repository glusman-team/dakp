package airflow

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/glusman-team/dakp/go/internal/dailymed"
)

const dailymedOperation = "extract_dailymed_spl"

// dailymedInterimOrder is the interim-table registration order; spl_documents is FIRST (the locked
// public contract — downstream shapers resolve "the dailymed parquet" via the first matching ref),
// mirroring spl_xml.SPLXMLExtractor.
var dailymedInterimOrder = []string{"spl_documents", "spl_sets", "spl_approvals", "spl_ingredients", "spl_sections"}

// ExtractDailyMed is the native Go implementation of the DAG's extract_dailymed stage. It mirrors
// spl_xml.SPLXMLExtractor._extract_via_go: parse the staged SPL inputs with internal/dailymed, then
// write the five normalized interim parquet tables (data/interim/dailymed/) plus the uncompressed
// sections TSV handoff (data/tabular/dailymed_spl_sections.tsv), registering each in the BLAKE3
// store. Returns the six ArtifactRefs in the same order as the Python extractor (5 parquet with
// spl_documents first, then the sections TSV).
func ExtractDailyMed(ctx context.Context, cfg Config, inputs []ArtifactRef) ([]ArtifactRef, error) {
	store := Store{Workdir: cfg.Workdir}
	interimDir := filepath.Join(store.InterimDir(), "dailymed")
	if err := os.MkdirAll(interimDir, 0o755); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(store.TabularDir(), 0o755); err != nil {
		return nil, err
	}

	// Stage the SPL inputs into a scratch dir (the parser classifies inputs by filename).
	scratch, err := os.MkdirTemp("", "dakp-dailymed-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(scratch)
	inDir := filepath.Join(scratch, "in")
	if err := StageInputs(inputs, inDir); err != nil {
		return nil, err
	}

	paths, err := dailymed.ListSPLFiles(inDir)
	if err != nil {
		return nil, err
	}
	if len(paths) == 0 {
		return nil, fmt.Errorf("extract_dailymed: no SPL files (.xml/.xml.gz) staged in %s", inDir)
	}

	limit := cfg.Threads
	if limit <= 0 {
		limit = 4
	}
	tables, err := dailymed.Extract(ctx, paths, limit)
	if err != nil {
		return nil, err
	}

	// Index the produced tables by base name (sans .tsv) for column/row lookup.
	byName := make(map[string]dailymed.TableFile, 5)
	for _, tf := range tables.TableFiles() {
		byName[strings.TrimSuffix(tf.Name, ".tsv")] = tf
	}

	inputIDs := tables.InputIDs
	warnings := int64(tables.Warnings)

	refs := make([]ArtifactRef, 0, len(dailymedInterimOrder)+1)
	for _, name := range dailymedInterimOrder {
		tf := byName[name]
		out := filepath.Join(interimDir, name+".parquet")
		n, err := WriteStringParquet(out, tf.Columns, tf.Rows)
		if err != nil {
			return nil, fmt.Errorf("extract_dailymed: write %s: %w", name, err)
		}
		ref, err := store.Register(RegisterInput{
			Path:              out,
			MediaType:         ParquetMediaType,
			Rows:              int64(n),
			SchemaFingerprint: SchemaFingerprint(tf.Columns),
			Inputs:            inputIDs,
			Warnings:          warnings,
			Operation:         dailymedOperation,
		})
		if err != nil {
			return nil, err
		}
		refs = append(refs, ref)
	}

	// Uncompressed sections TSV for the Tablassert handoff.
	secTF := byName["spl_sections"]
	tsvPath := filepath.Join(store.TabularDir(), "dailymed_spl_sections.tsv")
	if err := writeTableTSV(tsvPath, secTF.Columns, secTF.Rows); err != nil {
		return nil, fmt.Errorf("extract_dailymed: write sections tsv: %w", err)
	}
	tsvRef, err := store.Register(RegisterInput{
		Path:              tsvPath,
		MediaType:         TSVMediaType,
		Rows:              int64(len(secTF.Rows)),
		SchemaFingerprint: SchemaFingerprint(secTF.Columns),
		Inputs:            inputIDs,
		Warnings:          warnings,
		Operation:         dailymedOperation,
	})
	if err != nil {
		return nil, err
	}
	return append(refs, tsvRef), nil
}

// writeTableTSV writes columns + rows as an uncompressed TSV via the polars-compatible writer in
// internal/dailymed (byte-for-byte parity with Python polars write_csv(separator="\t")).
func writeTableTSV(path string, columns []string, rows [][]string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := dailymed.WriteTSV(f, columns, rows); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
