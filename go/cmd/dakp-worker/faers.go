package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/glusman-team/dakp/go/internal/blake3store"
	"github.com/glusman-team/dakp/go/internal/faers"
	"github.com/glusman-team/dakp/go/internal/registry"
)

// init self-registers the "faers" subcommand via the self-registration pattern (see
// hash.go and go/README.md): a new file in package main with an init() calling
// registry.Register. No other file (especially main.go) changes.
func init() {
	registry.Register("faers", func(ctx context.Context, args []string) error {
		return runFAERS(ctx, args, os.Stdout)
	})
}

// runFAERS implements the `faers` subcommand: parse a directory of FAERS ASCII `.txt`
// files (one or more quarters, derived from each filename), build the DELETE-filtered,
// cross-quarter-deduped case table, and write the uncompressed source-section TSVs
// (faers_cases.tsv + delete_audit.tsv + dedup_audit.tsv) to the output directory. The
// faers_cases.tsv columns match the Python schemas.FAERS_CASES_COLUMNS contract exactly.
//
// stdout receives only the b3:<hex> content id of faers_cases.tsv (machine-readable, like
// the hash subcommand); structured JSON logs go to stderr via log/slog.
func runFAERS(ctx context.Context, args []string, stdout io.Writer) error {
	fl := flag.NewFlagSet("faers", flag.ContinueOnError)
	fl.SetOutput(io.Discard) // keep stdout clean for the artifact id; errors are returned
	jobs := fl.Int("jobs", 4, "max concurrent file parses (errgroup SetLimit; <=0 = unbounded)")
	if err := fl.Parse(args); err != nil {
		return fmt.Errorf("faers: %w", err)
	}
	if fl.NArg() != 2 {
		return fmt.Errorf("faers: expected exactly two arguments: <quarter-dir> <out-dir>, got %d", fl.NArg())
	}
	inDir, outDir := fl.Arg(0), fl.Arg(1)

	logger := slog.New(slog.NewJSONHandler(os.Stderr, nil))
	start := time.Now()

	srcs, err := loadFAERSSources(inDir)
	if err != nil {
		return fmt.Errorf("faers: %w", err)
	}
	if len(srcs) == 0 {
		return fmt.Errorf("faers: no FAERS ASCII .txt files found under %s", inDir)
	}

	warn := &faers.Warnings{}
	res, err := faers.Extract(ctx, srcs, *jobs, warn)
	if err != nil {
		return fmt.Errorf("faers: %w", err)
	}

	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return fmt.Errorf("faers: %w", err)
	}
	casesPath := filepath.Join(outDir, "faers_cases.tsv")
	if err := writeTSVFile(casesPath, func(w io.Writer) error { return faers.WriteCasesTSV(w, res.Cases) }); err != nil {
		return fmt.Errorf("faers: write cases: %w", err)
	}
	// Audits: the Python emits these as parquet; Go has no parquet dep yet (go/README.md),
	// so they are written as uncompressed TSV with the same column contracts.
	deletePath := filepath.Join(outDir, "delete_audit.tsv")
	if err := writeTSVFile(deletePath, func(w io.Writer) error { return faers.WriteDeleteAuditTSV(w, res.DeleteAudit) }); err != nil {
		return fmt.Errorf("faers: write delete audit: %w", err)
	}
	dedupPath := filepath.Join(outDir, "dedup_audit.tsv")
	if err := writeTSVFile(dedupPath, func(w io.Writer) error { return faers.WriteDedupAuditTSV(w, res.DedupAudit) }); err != nil {
		return fmt.Errorf("faers: write dedup audit: %w", err)
	}

	id, err := blake3store.HashFile(casesPath)
	if err != nil {
		return fmt.Errorf("faers: hash cases: %w", err)
	}

	logger.Info("faers extract complete",
		"quarters", len(res.Quarters),
		"cases", len(res.Cases),
		"deleted", len(res.DeleteAudit),
		"deduped", len(res.DedupAudit),
		"warnings", warn.Total(),
		"artifact_id", id,
		"out_dir", outDir,
		"elapsed_ms", time.Since(start).Milliseconds(),
	)
	_, err = fmt.Fprintln(stdout, id)
	return err
}

// loadFAERSSources walks inDir (recursively) for `.txt` files, keeps those whose basename
// resolves to a FAERS family + quarter, and loads each as a faers.Source (content hashed
// for the b3-derived source_record_id). Files are processed in sorted path order for
// determinism. Mirrors faers_ascii._iter_faers_sources for loose .txt artifacts.
func loadFAERSSources(inDir string) ([]faers.Source, error) {
	var paths []string
	err := filepath.WalkDir(inDir, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			return nil
		}
		if strings.EqualFold(filepath.Ext(path), ".txt") {
			paths = append(paths, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(paths)

	var srcs []faers.Source
	for _, p := range paths {
		family, quarter := faers.FamilyAndQuarter(filepath.Base(p))
		if family == "" || quarter == "" {
			continue // not a FAERS ASCII file
		}
		content, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}
		srcs = append(srcs, faers.Source{
			Quarter:    quarter,
			Family:     family,
			Content:    content,
			SourceName: filepath.Base(p),
			SourceB3:   blake3store.HashBytes(content),
		})
	}
	return srcs, nil
}

// writeTSVFile creates path and streams the TSV produced by fn into it.
func writeTSVFile(path string, fn func(io.Writer) error) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := fn(f); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
