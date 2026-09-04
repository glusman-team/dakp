package airflow

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/glusman-team/dakp/go/internal/blake3store"
)

func TestRegisterRecordsSinglePassHashes(t *testing.T) {
	workdir := t.TempDir()
	p := filepath.Join(workdir, "out.tsv")
	if err := os.WriteFile(p, []byte("a\tb\n1\t2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	store := Store{Workdir: workdir}
	ref, err := store.Register(RegisterInput{Path: p, MediaType: TSVMediaType, Rows: 1})
	if err != nil {
		t.Fatalf("Register: %v", err)
	}

	wantID, err := blake3store.HashFile(p)
	if err != nil {
		t.Fatal(err)
	}
	wantSRI, err := blake3store.SHA256SRI(p)
	if err != nil {
		t.Fatal(err)
	}
	if ref.Blake3 != wantID {
		t.Errorf("ref.Blake3 = %q, want %q", ref.Blake3, wantID)
	}
	if ref.Manifest == nil {
		t.Fatal("ref.Manifest is nil")
	}
	m, err := blake3store.ReadManifest(*ref.Manifest)
	if err != nil {
		t.Fatalf("ReadManifest: %v", err)
	}
	if m.Hash.File == nil || *m.Hash.File != wantID {
		t.Errorf("manifest hash.file = %v, want %q", m.Hash.File, wantID)
	}
	if m.Hash.SHA256SRI == nil || *m.Hash.SHA256SRI != wantSRI {
		t.Errorf("manifest hash.sha256_sri = %v, want %q", m.Hash.SHA256SRI, wantSRI)
	}
}

// registerManyFixture writes n output files of varying content (one empty) into workdir
// and returns them as RegisterInputs with varied metadata.
func registerManyFixture(t *testing.T, workdir string, n int) []RegisterInput {
	t.Helper()
	inputs := make([]RegisterInput, 0, n)
	for i := 0; i < n; i++ {
		p := filepath.Join(workdir, "out"+string(rune('a'+i))+".tsv")
		var content []byte
		if i > 0 {
			content = []byte("a\tb\n" + strings.Repeat("1\t2\n", i*3))
		}
		if err := os.WriteFile(p, content, 0o644); err != nil {
			t.Fatal(err)
		}
		inputs = append(inputs, RegisterInput{
			Path:              p,
			MediaType:         TSVMediaType,
			Rows:              int64(i * 3),
			SchemaFingerprint: SchemaFingerprint([]string{"a", "b"}),
			Inputs:            []string{"b3:upstream"},
			Warnings:          int64(i),
			Operation:         "extract_test",
		})
	}
	return inputs
}

// TestRegisterManyMatchesSequentialRegister is the key equivalence test: RegisterMany on
// a batch must produce exactly the refs and manifest files that sequential Register calls
// produce on the same inputs.
func TestRegisterManyMatchesSequentialRegister(t *testing.T) {
	workdir := t.TempDir()
	inputs := registerManyFixture(t, workdir, 4)
	store := Store{Workdir: workdir}

	// Sequential oracle.
	var wantRefs []ArtifactRef
	wantManifests := map[string][]byte{}
	for _, in := range inputs {
		ref, err := store.Register(in)
		if err != nil {
			t.Fatalf("Register: %v", err)
		}
		wantRefs = append(wantRefs, ref)
		data, err := os.ReadFile(*ref.Manifest)
		if err != nil {
			t.Fatal(err)
		}
		wantManifests[*ref.Manifest] = data
	}

	// Wipe the manifests and re-register the same inputs as one batch.
	if err := os.RemoveAll(store.ManifestsDir()); err != nil {
		t.Fatal(err)
	}
	gotRefs, err := store.RegisterMany(context.Background(), inputs)
	if err != nil {
		t.Fatalf("RegisterMany: %v", err)
	}
	if len(gotRefs) != len(wantRefs) {
		t.Fatalf("RegisterMany returned %d refs, want %d", len(gotRefs), len(wantRefs))
	}
	for i := range wantRefs {
		if !reflect.DeepEqual(gotRefs[i], wantRefs[i]) {
			t.Errorf("ref[%d]: RegisterMany = %+v, sequential = %+v", i, gotRefs[i], wantRefs[i])
			continue
		}
		data, err := os.ReadFile(*gotRefs[i].Manifest)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(data, wantManifests[*gotRefs[i].Manifest]) {
			t.Errorf("ref[%d]: manifest bytes differ from sequential Register", i)
		}
	}
}

func TestRegisterManyEmpty(t *testing.T) {
	store := Store{Workdir: t.TempDir()}
	refs, err := store.RegisterMany(context.Background(), nil)
	if err != nil {
		t.Fatalf("RegisterMany(nil): %v", err)
	}
	if len(refs) != 0 {
		t.Errorf("RegisterMany(nil) returned %d refs, want 0", len(refs))
	}
}

func TestRegisterManyError(t *testing.T) {
	workdir := t.TempDir()
	inputs := registerManyFixture(t, workdir, 2)
	inputs = append(inputs, RegisterInput{Path: filepath.Join(workdir, "missing.tsv"), MediaType: TSVMediaType})
	store := Store{Workdir: workdir}
	if _, err := store.RegisterMany(context.Background(), inputs); err == nil {
		t.Fatal("expected error when one input path is missing")
	}
}
