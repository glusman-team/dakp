//go:build scale

// Scale verification for the bounded-memory streaming FAERS extract (plans/fix-faers-memory.md).
// Runs ONLY under the `scale` build tag against real FDA quarter zips, never in CI:
//
//	# streaming run over every zip in FAERS_SCALE_DIR (peak RSS via GNU time):
//	/usr/bin/time -v go test -tags scale ./internal/airflow/ -run TestExtractFAERSScaleStreaming -v -timeout 2h
//
//	# batch-vs-streaming byte parity over the (smaller) FAERS_SCALE_BATCH_DIR subset:
//	go test -tags scale ./internal/airflow/ -run TestExtractFAERSScaleParityVsBatch -v -timeout 2h
package airflow

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/glusman-team/dakp/go/internal/blake3store"
	"github.com/glusman-team/dakp/go/internal/faers"
)

// scaleZipInputs turns every .zip in dir into an ArtifactRef (sorted name order).
func scaleZipInputs(t *testing.T, dir string) []ArtifactRef {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read %s: %v", dir, err)
	}
	var names []string
	for _, e := range entries {
		if !e.IsDir() && strings.EqualFold(filepath.Ext(e.Name()), ".zip") {
			names = append(names, e.Name())
		}
	}
	if len(names) == 0 {
		t.Fatalf("no .zip files in %s", dir)
	}
	sort.Strings(names)
	var refs []ArtifactRef
	for _, n := range names {
		path := filepath.Join(dir, n)
		id, err := blake3store.HashFile(path)
		if err != nil {
			t.Fatalf("hash %s: %v", path, err)
		}
		refs = append(refs, ArtifactRef{URI: path, Blake3: id, MediaType: "application/zip"})
	}
	return refs
}

// reportMem logs heap stats after a forced GC (live-set proxy for the streaming path).
func reportMem(t *testing.T, label string) {
	t.Helper()
	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	t.Logf("%s: heap_live=%.2f GB total_alloc=%.2f GB", label, float64(m.HeapAlloc)/1e9, float64(m.TotalAlloc)/1e9)
}

func TestExtractFAERSScaleStreaming(t *testing.T) {
	dir := os.Getenv("FAERS_SCALE_DIR")
	if dir == "" {
		t.Skip("FAERS_SCALE_DIR not set")
	}
	inputs := scaleZipInputs(t, dir)
	cfg := Config{Workdir: t.TempDir(), Profile: "scale", Threads: 8}

	start := time.Now()
	refs, err := ExtractFAERS(context.Background(), cfg, inputs)
	if err != nil {
		t.Fatalf("ExtractFAERS: %v", err)
	}
	reportMem(t, "post-extract")

	if len(refs) != 5 {
		t.Fatalf("refs = %d, want 5", len(refs))
	}
	casesRows, tsvRows := *refs[0].Rows, *refs[1].Rows
	if casesRows != tsvRows {
		t.Fatalf("cases rows %d != tsv rows %d", casesRows, tsvRows)
	}
	tsvInfo, err := os.Stat(refs[1].URI)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("streaming extract: quarters=%d cases=%d deleted=%d deduped=%d tsv=%.2f GB elapsed=%s",
		len(inputs), casesRows, *refs[2].Rows, *refs[3].Rows, float64(tsvInfo.Size())/1e9, time.Since(start).Round(time.Millisecond))

	// Optional: copy the outputs to FAERS_SCALE_OUT for cross-run determinism checks.
	if out := os.Getenv("FAERS_SCALE_OUT"); out != "" {
		if err := os.MkdirAll(out, 0o755); err != nil {
			t.Fatal(err)
		}
		for i, name := range []string{"cases.parquet", "faers_cases.tsv"} {
			data, err := os.ReadFile(refs[i].URI)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(out, name), data, 0o644); err != nil {
				t.Fatal(err)
			}
		}
		t.Logf("outputs copied to %s", out)
	}
}

func TestExtractFAERSScaleParityVsBatch(t *testing.T) {
	dir := os.Getenv("FAERS_SCALE_BATCH_DIR")
	if dir == "" {
		t.Skip("FAERS_SCALE_BATCH_DIR not set")
	}
	inputs := scaleZipInputs(t, dir)

	// Streaming output.
	streamCfg := Config{Workdir: t.TempDir(), Profile: "scale", Threads: 8}
	streamRefs, err := ExtractFAERS(context.Background(), streamCfg, inputs)
	if err != nil {
		t.Fatalf("ExtractFAERS(stream): %v", err)
	}
	streamTSV, err := os.ReadFile(streamRefs[1].URI)
	if err != nil {
		t.Fatal(err)
	}

	// Batch oracle (the legacy all-in-memory path, same fixtures).
	srcs, err := loadFAERSSources(stageScaleInputs(t, inputs))
	if err != nil {
		t.Fatal(err)
	}
	res, err := faers.Extract(context.Background(), srcs, 8, &faers.Warnings{})
	if err != nil {
		t.Fatal(err)
	}
	var batchTSV bytes.Buffer
	if err := faers.WriteCasesTSV(&batchTSV, res.Cases); err != nil {
		t.Fatal(err)
	}

	if len(res.Cases) != int(*streamRefs[0].Rows) {
		t.Fatalf("batch cases %d != streaming cases %d", len(res.Cases), *streamRefs[0].Rows)
	}
	if !bytes.Equal(streamTSV, batchTSV.Bytes()) {
		t.Fatalf("streaming TSV (%d bytes) != batch TSV (%d bytes)", len(streamTSV), batchTSV.Len())
	}
	t.Logf("parity OK: %d cases, %d TSV bytes identical", len(res.Cases), len(streamTSV))
}

// stageScaleInputs hardlinks the scale zips into a fresh dir (StageInputs needs paths, and the
// batch loader walks a directory).
func stageScaleInputs(t *testing.T, refs []ArtifactRef) string {
	t.Helper()
	dir := t.TempDir()
	if err := StageInputs(refs, dir); err != nil {
		t.Fatal(err)
	}
	return dir
}
