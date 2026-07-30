package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/glusman-team/dakp/go/internal/blake3store"
	"github.com/glusman-team/dakp/go/internal/drugsfda"
	"github.com/glusman-team/dakp/go/internal/pipeline"
	"github.com/glusman-team/dakp/go/internal/registry"
)

// init self-registers the "drugsfda" subcommand via the same init()-registration pattern
// as hash.go: a new file in package main, no edits to main.go or any existing file.
func init() {
	registry.Register("drugsfda", func(ctx context.Context, args []string) error {
		return runDrugsFDA(ctx, args, os.Stdout)
	})
}

// drugsfdaInput is one recognized input file plus its classified table key.
type drugsfdaInput struct {
	path string
	key  string // products | applications | submissions
}

// drugsfdaTableOutput is the per-output-table provenance recorded in the stdout summary.
type drugsfdaTableOutput struct {
	Path              string `json:"path"`
	ArtifactID        string `json:"artifact_id"`
	Rows              int    `json:"rows"`
	MediaType         string `json:"media_type"`
	SchemaFingerprint string `json:"schema_fingerprint"`
}

// drugsfdaSummary is the machine-readable object written to stdout (the only stdout
// output; logs go to stderr as structured JSON).
type drugsfdaSummary struct {
	InputDir  string                         `json:"input_dir"`
	OutDir    string                         `json:"out_dir"`
	Inputs    map[string]string              `json:"inputs"`
	Tables    map[string]drugsfdaTableOutput `json:"tables"`
	Warnings  int                            `json:"warnings"`
	ElapsedMS int64                          `json:"elapsed_ms"`
}

// runDrugsFDA implements the `drugsfda` subcommand: read the Drugs@FDA tab-delimited
// inputs from <input-dir>, normalize them into products/applications/submissions/lookups
// tables, and write the uncompressed TSV source-section tables to <out-dir> (byte-compatible
// with the Python reference; see internal/drugsfda). stdout receives a JSON summary with
// each output's b3:<hex> artifact id; stderr receives structured JSON logs.
func runDrugsFDA(ctx context.Context, args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("drugsfda", flag.ContinueOnError)
	fs.SetOutput(io.Discard) // keep stdout clean for the JSON summary; errors are returned
	limit := fs.Int("limit", 4, "max concurrent input-file parses/hashes (<=0 = unbounded)")
	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("drugsfda: %w", err)
	}
	if fs.NArg() != 2 {
		return fmt.Errorf("drugsfda: expected <input-dir> <out-dir>, got %d argument(s)", fs.NArg())
	}
	inDir, outDir := fs.Arg(0), fs.Arg(1)

	log := slog.New(slog.NewJSONHandler(os.Stderr, nil)).With("task_id", "extract_drugsfda_products")
	start := time.Now()

	inputs, err := discoverDrugsFDAInputs(inDir)
	if err != nil {
		return fmt.Errorf("drugsfda: %w", err)
	}
	if len(inputs) == 0 {
		log.Warn("no Drugs@FDA tables recognized in inputs", "input_dir", inDir)
	}

	// Hash the recognized inputs (bounded concurrency via the blake3store foundation) for
	// the provenance chain, and parse them concurrently into Tables.
	paths := make([]string, len(inputs))
	for i, in := range inputs {
		paths[i] = in.path
	}
	inputHashes, err := blake3store.HashFiles(ctx, paths, *limit)
	if err != nil {
		return fmt.Errorf("drugsfda: hash inputs: %w", err)
	}

	tables, err := parseDrugsFDAInputs(ctx, inputs, *limit)
	if err != nil {
		return err
	}

	res := drugsfda.Extract(tables)
	for _, w := range res.Warnings {
		log.Warn("parse warning", "table", w.Table, "source_file", w.SourceFile, "message", w.Message)
	}

	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return fmt.Errorf("drugsfda: create out-dir: %w", err)
	}

	// Output tables in a fixed order; each is written only when its source table was
	// present (lookups derive from products), mirroring the Python extractor.
	specs := []struct {
		name    string
		columns []string
		rows    []drugsfda.Row
		present bool
	}{
		{"drugsfda_products.tsv", drugsfda.ProductsColumns, res.Products, res.HaveProducts},
		{"drugsfda_applications.tsv", drugsfda.ApplicationsColumns, res.Applications, res.HaveApplications},
		{"drugsfda_submissions.tsv", drugsfda.SubmissionsColumns, res.Submissions, res.HaveSubmissions},
		{"drugsfda_lookups.tsv", drugsfda.LookupsColumns, res.Lookups, res.HaveProducts},
	}

	summary := drugsfdaSummary{
		InputDir: inDir,
		OutDir:   outDir,
		Inputs:   make(map[string]string, len(inputs)),
		Tables:   make(map[string]drugsfdaTableOutput),
		Warnings: len(res.Warnings),
	}
	for _, in := range inputs {
		summary.Inputs[filepath.Base(in.path)] = inputHashes[in.path]
	}

	for _, spec := range specs {
		if !spec.present {
			continue
		}
		path := filepath.Join(outDir, spec.name)
		rows, err := drugsfda.WriteTSVFile(path, spec.columns, spec.rows)
		if err != nil {
			return fmt.Errorf("drugsfda: write %s: %w", spec.name, err)
		}
		id, err := blake3store.HashFile(path)
		if err != nil {
			return fmt.Errorf("drugsfda: hash %s: %w", spec.name, err)
		}
		summary.Tables[spec.name] = drugsfdaTableOutput{
			Path:              path,
			ArtifactID:        id,
			Rows:              rows,
			MediaType:         pipeline.InferMediaType(spec.name),
			SchemaFingerprint: blake3store.HashBytes([]byte(strings.Join(spec.columns, "\t"))),
		}
		log.Info("wrote source-section table", "table", spec.name, "artifact_id", id, "rows", rows)
	}

	summary.ElapsedMS = time.Since(start).Milliseconds()
	log.Info("extracted Drugs@FDA",
		"products", len(res.Products),
		"applications", len(res.Applications),
		"submissions", len(res.Submissions),
		"lookups", len(res.Lookups),
		"warnings", len(res.Warnings),
		"outputs", len(summary.Tables),
		"elapsed_ms", summary.ElapsedMS,
	)

	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(summary)
}

// discoverDrugsFDAInputs lists the regular files directly under inDir, classifies each by
// filename (drugsfda.Classify), and returns the recognized ones sorted by path for
// deterministic processing. Unrecognized files (e.g. SubmissionPropertyType.txt) are
// skipped.
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

// parseDrugsFDAInputs parses the recognized input files concurrently (errgroup with
// SetLimit for bounded concurrency, cancelling on first error) and assembles them into a
// drugsfda.Tables. Results are stored by index so concurrency never affects ordering; the
// Tables are then filled in sorted path order (last file wins per key, mirroring the
// Python collector's dict assignment).
func parseDrugsFDAInputs(ctx context.Context, inputs []drugsfdaInput, limit int) (drugsfda.Tables, error) {
	results := make([]drugsfda.Table, len(inputs))
	g, gctx := errgroup.WithContext(ctx)
	if limit > 0 {
		g.SetLimit(limit)
	}
	for i, in := range inputs {
		g.Go(func() error {
			if err := gctx.Err(); err != nil {
				return err
			}
			tbl, err := drugsfda.ParseTSV(in.path)
			if err != nil {
				return fmt.Errorf("drugsfda: %w", err)
			}
			results[i] = tbl
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return drugsfda.Tables{}, err
	}

	var tables drugsfda.Tables
	for i, in := range inputs {
		tbl := results[i]
		switch in.key {
		case "products":
			tables.Products = &tbl
		case "applications":
			tables.Applications = &tbl
		case "submissions":
			tables.Submissions = &tbl
		}
	}
	return tables, nil
}
