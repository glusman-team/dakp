package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"time"

	"github.com/glusman-team/dakp/go/internal/blake3store"
	"github.com/glusman-team/dakp/go/internal/dailymed"
	"github.com/glusman-team/dakp/go/internal/registry"
)

// init self-registers the "dailymed" subcommand via the same self-registration pattern as
// hash.go: a new file in package main with an init() calling registry.Register. No other
// file (especially main.go) changes.
func init() {
	registry.Register("dailymed", func(ctx context.Context, args []string) error {
		return runDailyMed(ctx, args, os.Stdout)
	})
}

// runDailyMed implements the `dailymed` subcommand: extract a directory/shard of gzipped
// (or plain) SPL XML into the five normalized, uncompressed TSV source tables (documents,
// sets, approvals, ingredients, sections) written to an output directory. The column
// layout and source_record_id derivation match the Python extractor
// (src/dakp_pipeline/extract/spl_xml.py) for cross-language parity.
//
// Usage: dakp-worker dailymed [-limit N] <input-dir> <output-dir>
//
// stdout receives the single machine-readable result: the b3:<hex> tree hash of the output
// directory (the canonical artifact id of the produced tables). Structured JSON logs go to
// stderr via log/slog.
func runDailyMed(ctx context.Context, args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("dailymed", flag.ContinueOnError)
	fs.SetOutput(io.Discard) // keep stdout clean for the artifact id; errors are returned
	limit := fs.Int("limit", 4, "max SPL files parsed concurrently (<=0 = unbounded)")
	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("dailymed: %w", err)
	}
	if fs.NArg() != 2 {
		return fmt.Errorf("dailymed: expected exactly two arguments <input-dir> <output-dir>, got %d", fs.NArg())
	}
	inDir, outDir := fs.Arg(0), fs.Arg(1)

	logger := slog.New(slog.NewJSONHandler(os.Stderr, nil))
	start := time.Now()

	paths, err := dailymed.ListSPLFiles(inDir)
	if err != nil {
		return fmt.Errorf("dailymed: %w", err)
	}
	if len(paths) == 0 {
		return fmt.Errorf("dailymed: no SPL files (.xml/.xml.gz) found in %s", inDir)
	}

	tables, err := dailymed.Extract(ctx, paths, nil, *limit)
	if err != nil {
		return err
	}
	if err := tables.WriteDir(outDir); err != nil {
		return fmt.Errorf("dailymed: write %s: %w", outDir, err)
	}

	// Canonical artifact id for the whole extraction: the deterministic tree hash of the
	// produced tables (byte-stable for identical output).
	outHash, err := blake3store.HashTree(outDir)
	if err != nil {
		return fmt.Errorf("dailymed: hash output %s: %w", outDir, err)
	}

	logger.Info("extracted dailymed SPL",
		"task_id", "extract_dailymed_spl",
		"input_dir", inDir,
		"output_dir", outDir,
		"input_files", len(paths),
		"input_ids", tables.InputIDs,
		"documents", len(tables.Documents),
		"sets", len(tables.Sets),
		"approvals", len(tables.Approvals),
		"ingredients", len(tables.Ingredients),
		"sections", len(tables.Sections),
		"warnings", tables.Warnings,
		"output_hash", outHash,
		"elapsed_ms", time.Since(start).Milliseconds(),
	)

	_, err = fmt.Fprintln(stdout, outHash)
	return err
}
