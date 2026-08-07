package airflow

import (
	"archive/zip"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/glusman-team/dakp/go/internal/drugsfda"
)

const drugsfdaWarningsMediaType = "application/x-ndjson"

// drugsfdaEvent prefixes every stat line the Drugs@FDA extractor emits (one stat per line).
const drugsfdaEvent = "extract_drugsfda"

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
	Stat(ctx, drugsfdaEvent, "staged_inputs", len(inputs))
	if err := unpackDrugsFDAZips(inDir); err != nil {
		return nil, err
	}

	discovered, err := discoverDrugsFDAInputs(inDir)
	if err != nil {
		return nil, err
	}
	Stat(ctx, drugsfdaEvent, "inputs_discovered", len(discovered))
	for _, in := range discovered {
		StatDebug(ctx, drugsfdaEvent, "input", in.key+" <- "+filepath.Base(in.path))
	}
	tables, err := parseDrugsFDAInputs(discovered)
	if err != nil {
		return nil, err
	}
	res := drugsfda.Extract(tables)
	Stat(ctx, drugsfdaEvent, "parse_warnings", len(res.Warnings))
	warnings := int64(len(res.Warnings))
	inputIDs := make([]string, len(inputs))
	for i, ref := range inputs {
		inputIDs[i] = ref.Blake3
	}

	var refs []ArtifactRef
	if res.HaveProducts {
		Stat(ctx, drugsfdaEvent, "parsed_products_rows", len(res.Products))
		ref, err := writeDrugsFDAParquet(store, drugsfdaDir, "products", drugsfda.ProductsColumns, res.Products, inputIDs, warnings)
		if err != nil {
			return nil, err
		}
		StatOutput(ctx, drugsfdaEvent, "products.parquet", ref)
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
		StatOutput(ctx, drugsfdaEvent, "drugsfda_products.tsv", tsvRef)
		refs = append(refs, tsvRef)
	}
	if res.HaveApplications {
		Stat(ctx, drugsfdaEvent, "parsed_applications_rows", len(res.Applications))
		ref, err := writeDrugsFDAParquet(store, drugsfdaDir, "applications", drugsfda.ApplicationsColumns, res.Applications, inputIDs, warnings)
		if err != nil {
			return nil, err
		}
		StatOutput(ctx, drugsfdaEvent, "applications.parquet", ref)
		refs = append(refs, ref)
	}
	if res.HaveSubmissions {
		Stat(ctx, drugsfdaEvent, "parsed_submissions_rows", len(res.Submissions))
		ref, err := writeDrugsFDAParquet(store, drugsfdaDir, "submissions", drugsfda.SubmissionsColumns, res.Submissions, inputIDs, warnings)
		if err != nil {
			return nil, err
		}
		StatOutput(ctx, drugsfdaEvent, "submissions.parquet", ref)
		refs = append(refs, ref)
	}
	if res.HaveProducts { // lookups derive from products
		Stat(ctx, drugsfdaEvent, "derived_lookups_rows", len(res.Lookups))
		ref, err := writeDrugsFDAParquet(store, drugsfdaDir, "lookups", drugsfda.LookupsColumns, res.Lookups, inputIDs, 0)
		if err != nil {
			return nil, err
		}
		StatOutput(ctx, drugsfdaEvent, "lookups.parquet", ref)
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
	StatOutput(ctx, drugsfdaEvent, "extract_warnings.jsonl", warningsRef)

	Stat(ctx, drugsfdaEvent, "output_refs", len(refs)+1)
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

// unpackDrugsFDAZips extracts the members of any .zip files directly under inDir into inDir (flat,
// basename only), so the Drugs@FDA ASCII tables (Products.txt / Applications.txt / Submissions.txt)
// that ship inside the downloaded data-files zip become discoverable as loose files. Loose tables
// (the fixture mirrors) are left untouched. Mirrors the Python collector, which reads the tables
// straight out of the zip.
func unpackDrugsFDAZips(inDir string) error {
	entries, err := os.ReadDir(inDir)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if e.IsDir() || !strings.EqualFold(filepath.Ext(e.Name()), ".zip") {
			continue
		}
		if err := extractZipFlat(filepath.Join(inDir, e.Name()), inDir); err != nil {
			return err
		}
	}
	return nil
}

// extractZipFlat writes each non-directory member of zipPath to destDir under its basename.
func extractZipFlat(zipPath, destDir string) error {
	reader, err := zip.OpenReader(zipPath)
	if err != nil {
		return err
	}
	defer reader.Close()
	for _, f := range reader.File {
		if f.FileInfo().IsDir() {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		dst, err := os.Create(filepath.Join(destDir, filepath.Base(f.Name)))
		if err != nil {
			rc.Close()
			return err
		}
		_, err = io.Copy(dst, rc)
		dst.Close()
		rc.Close()
		if err != nil {
			return err
		}
	}
	return nil
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
