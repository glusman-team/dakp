package airflow

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// opIndexVectorInputs are the shared Python/Go key test-vector inputs (deliberately unsorted;
// both implementations sort before hashing).
var opIndexVectorInputs = []string{
	"b3:" + strings.Repeat("b", 64),
	"b3:" + strings.Repeat("a", 64),
}

// TestOpIndexKeyMatchesPythonVector pins the cross-language key spec (see the spec comment in
// opindex.go): the Python artifact_store.op_index_key test asserts the same constants, so the
// two implementations can never drift silently.
func TestOpIndexKeyMatchesPythonVector(t *testing.T) {
	// python: op_index_key("extract_dailymed", ["b3:"+"b"*64, "b3:"+"a"*64])
	if got := OpIndexKey("extract_dailymed", opIndexVectorInputs); got != "4f10631218994be14487284cf116f4ae374a0fc99ad572926a67436f77dd1ad6" {
		t.Errorf("OpIndexKey(extract_dailymed) = %q, want Python vector", got)
	}
	// python: op_index_key("shape_approved_treats", ["b3:"+"c"*64, "b3:"+"d"*64, "b3:"+"e"*64])
	shapeInputs := []string{"b3:" + strings.Repeat("c", 64), "b3:" + strings.Repeat("d", 64), "b3:" + strings.Repeat("e", 64)}
	if got := OpIndexKey("shape_approved_treats", shapeInputs); got != "4c104fb4bd812ccdec9fb799a96e5c1209f2c5876eb064634157b001d533d644" {
		t.Errorf("OpIndexKey(shape_approved_treats) = %q, want Python vector", got)
	}
	// Input order must not matter.
	reversed := []string{opIndexVectorInputs[1], opIndexVectorInputs[0]}
	if OpIndexKey("extract_dailymed", reversed) != OpIndexKey("extract_dailymed", opIndexVectorInputs) {
		t.Error("OpIndexKey is input-order sensitive")
	}
}

// registerOpTestFile writes a real output file and registers it (creating its manifest), so
// operation-index lookups have store-real artifacts to verify against.
func registerOpTestFile(t *testing.T, store Store, dir, name, content, operation string, inputs []string) ArtifactRef {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	ref, err := store.Register(RegisterInput{Path: p, MediaType: TSVMediaType, Rows: 1, Inputs: inputs, Operation: operation})
	if err != nil {
		t.Fatalf("Register %s: %v", name, err)
	}
	return ref
}

func TestRecordAndFindByOperationRoundTrip(t *testing.T) {
	workdir := t.TempDir()
	store := Store{Workdir: workdir}
	outDir := t.TempDir()
	inputs := []string{"b3:" + strings.Repeat("1", 64), "b3:" + strings.Repeat("2", 64)}
	refA := registerOpTestFile(t, store, outDir, "a.tsv", "a\n", "extract_x_a", inputs)
	refB := registerOpTestFile(t, store, outDir, "b.tsv", "b\n", "extract_x_b", inputs)

	store.RecordOperation("extract_x", inputs, []ArtifactRef{refA, refB})

	got := store.FindByOperation("extract_x", inputs)
	if len(got) != 2 {
		t.Fatalf("FindByOperation returned %d refs, want 2", len(got))
	}
	for i, want := range []ArtifactRef{refA, refB} {
		if got[i].URI != want.URI || got[i].Blake3 != want.Blake3 || got[i].MediaType != want.MediaType {
			t.Errorf("ref[%d] = %+v, want %+v", i, got[i], want)
		}
		if got[i].Rows == nil || *got[i].Rows != 1 {
			t.Errorf("ref[%d].Rows = %v, want 1", i, got[i].Rows)
		}
		if got[i].Manifest == nil || *got[i].Manifest != store.ManifestPath(want.Blake3) {
			t.Errorf("ref[%d].Manifest = %v", i, got[i].Manifest)
		}
	}

	// Re-record replaces the entry wholesale (a re-run publishes the new artifact id).
	refA2 := registerOpTestFile(t, store, outDir, "a.tsv", "a v2\n", "extract_x_a", inputs)
	store.RecordOperation("extract_x", inputs, []ArtifactRef{refA2, refB})
	got = store.FindByOperation("extract_x", inputs)
	if len(got) != 2 || got[0].Blake3 != refA2.Blake3 {
		t.Errorf("after re-record got %+v, want first ref %q", got, refA2.Blake3)
	}
}

func TestFindByOperationMisses(t *testing.T) {
	workdir := t.TempDir()
	store := Store{Workdir: workdir}
	outDir := t.TempDir()
	inputs := []string{"b3:" + strings.Repeat("1", 64)}
	ref := registerOpTestFile(t, store, outDir, "a.tsv", "a\n", "extract_x_a", inputs)
	store.RecordOperation("extract_x", inputs, []ArtifactRef{ref})

	if got := store.FindByOperation("extract_x", []string{"b3:" + strings.Repeat("9", 64)}); got != nil {
		t.Errorf("changed inputs must miss, got %+v", got)
	}
	if got := store.FindByOperation("other_op", inputs); got != nil {
		t.Errorf("unknown operation must miss, got %+v", got)
	}
	if got := store.FindByOperation("extract_x", nil); got != nil {
		t.Errorf("empty inputs must miss, got %+v", got)
	}
	// A corrupt index file reads as empty (the next record rewrites it fresh).
	if err := os.WriteFile(store.opIndexPath(), []byte("not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := store.FindByOperation("extract_x", inputs); got != nil {
		t.Errorf("corrupt index must miss, got %+v", got)
	}
}

func TestFindByOperationPrunesStaleEntry(t *testing.T) {
	workdir := t.TempDir()
	store := Store{Workdir: workdir}
	outDir := t.TempDir()
	inputs := []string{"b3:" + strings.Repeat("1", 64)}
	ref := registerOpTestFile(t, store, outDir, "a.tsv", "a\n", "extract_x_a", inputs)
	store.RecordOperation("extract_x", inputs, []ArtifactRef{ref})

	// Deleting the artifact file turns the hit into a miss and prunes the entry.
	if err := os.Remove(ref.URI); err != nil {
		t.Fatal(err)
	}
	if got := store.FindByOperation("extract_x", inputs); got != nil {
		t.Fatalf("deleted artifact must miss, got %+v", got)
	}
	if _, ok := store.readOpIndex().Entries[OpIndexKey("extract_x", inputs)]; ok {
		t.Error("stale entry was not pruned")
	}

	// Same for a deleted manifest.
	ref2 := registerOpTestFile(t, store, outDir, "a.tsv", "a v2\n", "extract_x_a", inputs)
	store.RecordOperation("extract_x", inputs, []ArtifactRef{ref2})
	if err := os.Remove(store.ManifestPath(ref2.Blake3)); err != nil {
		t.Fatal(err)
	}
	if got := store.FindByOperation("extract_x", inputs); got != nil {
		t.Fatalf("deleted manifest must miss, got %+v", got)
	}
}

// TestFindByOperationReadsPythonIndex hand-writes the index JSON in the exact shape Python's
// ArtifactStore emits (json.dumps(indent=2)) to prove Go reads Python-written entries.
func TestFindByOperationReadsPythonIndex(t *testing.T) {
	workdir := t.TempDir()
	store := Store{Workdir: workdir}
	outDir := t.TempDir()
	inputs := opIndexVectorInputs
	ref := registerOpTestFile(t, store, outDir, "spl_sections.parquet", "x\n", "extract_dailymed_spl", inputs)

	// The Python writer's JSON: 2-space indent, entry keyed by op_index_key(operation, inputs).
	pythonJSON := fmt.Sprintf(`{
  "version": 1,
  "entries": {
    "%s": {
      "operation": "extract_dailymed",
      "inputs": [
        "%s",
        "%s"
      ],
      "outputs": [
        {
          "artifact_id": "%s",
          "path": "%s",
          "media_type": "%s",
          "rows": 1,
          "schema_fingerprint": null
        }
      ]
    }
  }
}`, OpIndexKey("extract_dailymed", inputs), opIndexVectorInputs[1], opIndexVectorInputs[0], ref.Blake3, ref.URI, ref.MediaType)
	if err := os.MkdirAll(store.ManifestsDir(), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(store.opIndexPath(), []byte(pythonJSON), 0o644); err != nil {
		t.Fatal(err)
	}

	got := store.FindByOperation("extract_dailymed", inputs)
	if len(got) != 1 || got[0].Blake3 != ref.Blake3 || got[0].URI != ref.URI {
		t.Fatalf("FindByOperation over Python-written index = %+v, want %q", got, ref.Blake3)
	}
	if got[0].Rows == nil || *got[0].Rows != 1 {
		t.Errorf("rows = %v, want 1", got[0].Rows)
	}
	if got[0].SchemaFingerprint != nil {
		t.Errorf("schema_fingerprint = %v, want nil", *got[0].SchemaFingerprint)
	}
}

// TestExtractDrugsFDASkipsWhenUnchanged covers the task-level skip: a second extract over the
// same upstream refs returns the cached refs WITHOUT re-extracting (proven by deleting the
// input file — staging it would fail), while force re-extracts (and fails on the deleted input).
func TestExtractDrugsFDASkipsWhenUnchanged(t *testing.T) {
	// Copy the shared fixtures so the quarantine move below never touches testdata.
	refs := drugsfdaFixtureRefs(t)
	inputDir := t.TempDir()
	for i, ref := range refs {
		dest := filepath.Join(inputDir, filepath.Base(ref.URI))
		data, err := os.ReadFile(ref.URI)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(dest, data, 0o644); err != nil {
			t.Fatal(err)
		}
		refs[i].URI = dest
	}
	workdir := t.TempDir()
	cfg := Config{Workdir: workdir, Profile: "mock", Threads: 4}
	first, err := ExtractDrugsFDA(context.Background(), cfg, refs)
	if err != nil {
		t.Fatalf("ExtractDrugsFDA: %v", err)
	}

	// Move the inputs away but keep the ORIGINAL (now dangling) URIs in the refs: any real
	// re-extraction must fail staging them, so a successful second run proves the skip fired.
	quarantine := t.TempDir()
	moved := make([]ArtifactRef, len(refs))
	for i, ref := range refs {
		if err := os.Rename(ref.URI, filepath.Join(quarantine, filepath.Base(ref.URI))); err != nil {
			t.Fatal(err)
		}
		moved[i] = ref
	}

	cached, err := ExtractDrugsFDA(context.Background(), cfg, moved)
	if err != nil {
		t.Fatalf("ExtractDrugsFDA(skip): %v", err)
	}
	if len(cached) != len(first) {
		t.Fatalf("skip returned %d refs, want %d", len(cached), len(first))
	}
	for i := range first {
		if cached[i].URI != first[i].URI || cached[i].Blake3 != first[i].Blake3 {
			t.Errorf("cached ref[%d] = %+v, want identical to first run %+v", i, cached[i], first[i])
		}
	}

	// force bypasses the skip and really re-extracts (failing on the moved-away inputs).
	cfgForce := Config{Workdir: workdir, Profile: "mock", Threads: 4, Force: true}
	if _, err := ExtractDrugsFDA(context.Background(), cfgForce, moved); err == nil {
		t.Error("force run over deleted inputs must fail (proving no skip), got nil error")
	}

	// A changed input id set must also miss the cache.
	other := append([]ArtifactRef{}, moved...)
	other[0].Blake3 = "b3:" + strings.Repeat("0", 64)
	if _, err := ExtractDrugsFDA(context.Background(), cfg, other); err == nil {
		t.Error("changed inputs must not skip (and fail staging), got nil error")
	}
}
