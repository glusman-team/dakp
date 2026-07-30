package airflow

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/glusman-team/dakp/go/internal/drugsfda"
)

const drugsfdaWarningsMediaType = "application/x-ndjson"

// drugsfdaInput is one recognized input file plus its classified table key.
type drugsfdaInput struct {
	path string
	key  string // products | applications | submissions
}

// ExtractDrugsFDA is the native Go implementation of the DAG's extract_drugsfda stage. It mirrors
// drugsfda_products.DrugsFDAProductsExtractor._extract_via_go: read the staged Drugs@FDA
// tab-delimited tables with internal/drugsfda, then register the products/applications/submissions/
// lookups interim parquets (each only when its source table is present; lookups derive from
// products), the public drugsfda_products.tsv handoff, and an empty extract_warnings.jsonl. Refs are
// returned in the same order as the Python extractor.
func ExtractDrugsFDA(ctx context.Context, cfg Config, inputs []ArtifactRef) ([]ArtifactRef, error) {
	store := Store{Workdir: cfg.Workdir}
	drugsfdaDir := filepath.Join(store.InterimDir(), "drugsfda")
	if err := os.MkdirAll(drugsfdaDir, 0o755); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(store.TabularDir(), 0o755); err != nil {
		return nil, err
	}

	scratch, err := os.MkdirTemp("", "dakp-drugsfda-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(scratch)
	inDir := filepath.Join(scratch, "in")
	if err := StageInputs(inputs, inDir); err != nil {
		return nil, err
	}

	discovered, err := discoverDrugsFDAInputs(inDir)
	if err != nil {
		return nil, err
	}
	tables, err := parseDrugsFDAInputs(discovered)
	if err != nil {
		return nil, err
	}
	res := drugsfda.Extract(tables)
	warnings := int64(len(res.Warnings))
	inputIDs := make([]string, len(inputs))
	for i, ref := range inputs {
		inputIDs[i] = ref.Blake3
	}

	var refs []ArtifactRef
	if res.HaveProducts {
		ref, err := writeDrugsFDAParquet(store, drugsfdaDir, "products", drugsfda.ProductsColumns, res.Products, inputIDs, warnings)
		if err != nil {
			return nil, err
		}
		refs = append(refs, ref)

		// Public drugsfda_products.tsv (Tablassert source-section handoff).
		tsvPath := filepath.Join(store.TabularDir(), "drugsfda_products.tsv")
		tsvRows, err := drugsfda.WriteTSVFile(tsvPath, drugsfda.ProductsColumns, res.Products)
		if err != nil {
			return nil, fmt.Errorf("extract_drugsfda: write products tsv: %w", err)
		}
		tsvRef, err := store.Register(RegisterInput{
			Path: tsvPath, MediaType: TSVMediaType, Rows: int64(tsvRows),
			SchemaFingerprint: SchemaFingerprint(drugsfda.ProductsColumns), Inputs: inputIDs,
			Warnings: warnings, Operation: "emit_drugsfda_products_tsv",
		})
		if err != nil {
			return nil, err
		}
		refs = append(refs, tsvRef)
	}
	if res.HaveApplications {
		ref, err := writeDrugsFDAParquet(store, drugsfdaDir, "applications", drugsfda.ApplicationsColumns, res.Applications, inputIDs, warnings)
		if err != nil {
			return nil, err
		}
		refs = append(refs, ref)
	}
	if res.HaveSubmissions {
		ref, err := writeDrugsFDAParquet(store, drugsfdaDir, "submissions", drugsfda.SubmissionsColumns, res.Submissions, inputIDs, warnings)
		if err != nil {
			return nil, err
		}
		refs = append(refs, ref)
	}
	if res.HaveProducts { // lookups derive from products
		ref, err := writeDrugsFDAParquet(store, drugsfdaDir, "lookups", drugsfda.LookupsColumns, res.Lookups, inputIDs, 0)
		if err != nil {
			return nil, err
		}
		refs = append(refs, ref)
	}

	// Parse-warning provenance record (kept empty; warnings are in the task log stream).
	warningsPath := filepath.Join(drugsfdaDir, "extract_warnings.jsonl")
	if err := os.WriteFile(warningsPath, nil, 0o644); err != nil {
		return nil, err
	}
	warningsRef, err := store.Register(RegisterInput{
		Path: warningsPath, MediaType: drugsfdaWarningsMediaType, Rows: 0, Inputs: inputIDs,
		Operation: "extract_drugsfda_warnings",
	})
	if err != nil {
		return nil, err
	}
	return append(refs, warningsRef), nil
}

func writeDrugsFDAParquet(store Store, dir, name string, columns []string, rows []drugsfda.Row, inputIDs []string, warnings int64) (ArtifactRef, error) {
	path := filepath.Join(dir, name+".parquet")
	grid := make([][]string, len(rows))
	for i, row := range rows {
		rec := make([]string, len(columns))
		for ci, col := range columns {
			rec[ci] = row[col]
		}
		grid[i] = rec
	}
	n, err := WriteStringParquet(path, columns, grid)
	if err != nil {
		return ArtifactRef{}, fmt.Errorf("extract_drugsfda: write %s: %w", name, err)
	}
	return store.Register(RegisterInput{
		Path: path, MediaType: ParquetMediaType, Rows: int64(n),
		SchemaFingerprint: SchemaFingerprint(columns), Inputs: inputIDs,
		Warnings: warnings, Operation: "extract_drugsfda_" + name,
	})
}

// discoverDrugsFDAInputs lists the regular files directly under inDir, classifies each by filename
// (drugsfda.Classify), and returns the recognized ones sorted by path for deterministic processing.
func discoverDrugsFDAInputs(inDir string) ([]drugsfdaInput, error) {
	entries, err := os.ReadDir(inDir)
	if err != nil {
		return nil, err
	}
	var inputs []drugsfdaInput
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		key := drugsfda.Classify(e.Name())
		if key == "" {
			continue
		}
		inputs = append(inputs, drugsfdaInput{path: filepath.Join(inDir, e.Name()), key: key})
	}
	sort.Slice(inputs, func(i, j int) bool { return inputs[i].path < inputs[j].path })
	return inputs, nil
}

// parseDrugsFDAInputs parses the recognized inputs in sorted path order (last file wins per key,
// mirroring the Python collector's dict assignment) and assembles a drugsfda.Tables.
func parseDrugsFDAInputs(inputs []drugsfdaInput) (drugsfda.Tables, error) {
	var tables drugsfda.Tables
	for _, in := range inputs {
		tbl, err := drugsfda.ParseTSV(in.path)
		if err != nil {
			return tables, fmt.Errorf("extract_drugsfda: %w", err)
		}
		parsed := tbl
		switch in.key {
		case "products":
			tables.Products = &parsed
		case "applications":
			tables.Applications = &parsed
		case "submissions":
			tables.Submissions = &parsed
		}
	}
	return tables, nil
}
