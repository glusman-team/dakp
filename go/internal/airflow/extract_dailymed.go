package airflow

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/glusman-team/dakp/go/internal/dailymed"
)

const dailymedOperation = "extract_dailymed_spl"

// dailymedTaskOperation is the task-level operation-index key for the already-done skip
// (the per-artifact manifests keep dailymedOperation; see opindex.go).
const dailymedTaskOperation = "extract_dailymed"

// dailymedEvent prefixes every stat line the DailyMed extractor emits (one stat per line).
const dailymedEvent = "extract_dailymed"

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
	upstreamIDs := upstreamInputIDs(inputs)
	if !cfg.Force {
		// Already-done skip: the same upstream artifacts were extracted before.
		if cached := store.FindByOperation(dailymedTaskOperation, upstreamIDs); cached != nil {
			Stat(ctx, dailymedEvent, "skipped_already_done", len(cached))
			return cached, nil
		}
	}
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
	Stat(ctx, dailymedEvent, "staged_inputs", len(inputs))

	paths, err := dailymed.ListSPLFiles(inDir)
	if err != nil {
		return nil, err
	}
	if len(paths) == 0 {
		return nil, fmt.Errorf("extract_dailymed: no SPL files (.xml/.xml.gz) staged in %s", inDir)
	}
	Stat(ctx, dailymedEvent, "spl_files", len(paths))

	limit := cfg.Threads
	if limit <= 0 {
		limit = 4
	}
	Stat(ctx, dailymedEvent, "workers", limit)
	// Reuse the input refs' BLAKE3 ids (carried over XCom from acquisition) instead of
	// re-hashing every SPL file; Extract hashes any staged path whose id is unknown.
	idsByName := make(map[string]string, len(inputs))
	for _, ref := range inputs {
		if ref.Blake3 != "" {
			idsByName[filepath.Base(ref.URI)] = ref.Blake3
		}
	}
	ids := make([]string, len(paths))
	for i, p := range paths {
		name := filepath.Base(p)
		if id, ok := idsByName[name]; ok {
			ids[i] = id
			continue
		}
		// StageInputs renames basename collisions to NNNN_<base>; strip that prefix.
		if len(name) > 5 && name[4] == '_' {
			if _, convErr := strconv.Atoi(name[:4]); convErr == nil {
				if id, ok := idsByName[name[5:]]; ok {
					ids[i] = id
				}
			}
		}
	}
	parseStart := time.Now()
	tables, err := dailymed.Extract(ctx, paths, ids, limit)
	if err != nil {
		return nil, err
	}
	Stat(ctx, dailymedEvent, "parse_elapsed_s", fmt.Sprintf("%.3f", time.Since(parseStart).Seconds()))
	Stat(ctx, dailymedEvent, "parse_warnings", tables.Warnings)

	// Index the produced tables by base name (sans .tsv) for column/row lookup.
	byName := make(map[string]dailymed.TableFile, 5)
	for _, tf := range tables.TableFiles() {
		byName[strings.TrimSuffix(tf.Name, ".tsv")] = tf
	}
	for _, name := range dailymedInterimOrder {
		Stat(ctx, dailymedEvent, "parsed_"+name+"_rows", len(byName[name].Rows))
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
		StatOutput(ctx, dailymedEvent, name+".parquet", ref)
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
	StatOutput(ctx, dailymedEvent, "dailymed_spl_sections.tsv", tsvRef)

	refs = append(refs, tsvRef)
	Stat(ctx, dailymedEvent, "output_refs", len(refs))
	store.RecordOperation(dailymedTaskOperation, upstreamIDs, refs)
	return refs, nil
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
