package blake3store

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestManifestByteParityWithPython asserts that Go re-marshals the Python-generated golden
// fixtures byte-for-byte. The fixtures under testdata/ were written by Python's
// ArtifactManifest.model_dump_json(indent=2); reading them into Go and re-marshalling must
// reproduce the exact same bytes, proving the manifest shape (field order, nullability,
// 2-space indent, empty inputs as []) is identical across languages.
func TestManifestByteParityWithPython(t *testing.T) {
	for _, name := range []string{"manifest_full.json", "manifest_minimal.json"} {
		path := filepath.Join("testdata", name)
		wantBytes, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}

		m, err := ReadManifest(path)
		if err != nil {
			t.Fatalf("%s: ReadManifest: %v", name, err)
		}
		gotBytes, err := m.Marshal()
		if err != nil {
			t.Fatalf("%s: Marshal: %v", name, err)
		}
		if !bytes.Equal(gotBytes, wantBytes) {
			t.Errorf("%s: Go marshal != Python bytes\n--- got ---\n%s\n--- want ---\n%s",
				name, gotBytes, wantBytes)
		}
	}
}

func TestManifestRoundTrip(t *testing.T) {
	m := NewArtifactManifest("b3:"+strings.Repeat("ab", 32), "data/tabular/x.tsv", "text/tab-separated-values")
	m.Hash.File = StringPtr(m.ArtifactID)
	m.Hash.SHA256SRI = StringPtr("sha256-xyz=")
	m.Inputs = []string{"b3:" + strings.Repeat("11", 32)}
	m.Operation = &OperationBlock{Name: "shape_x", Version: "v1", ConfigHash: StringPtr("b3:" + strings.Repeat("22", 32))}
	m.Source.URL = StringPtr("https://example.invalid/x.tsv")
	m.Environment.GitCommit = StringPtr(strings.Repeat("0", 40))
	m.Table.Rows = IntPtr(42)
	m.Table.Warnings = IntPtr(3)

	dir := t.TempDir()
	p := filepath.Join(dir, "nested", "manifest.json")
	if err := WriteManifest(p, m); err != nil {
		t.Fatalf("WriteManifest: %v", err)
	}
	back, err := ReadManifest(p)
	if err != nil {
		t.Fatalf("ReadManifest: %v", err)
	}

	if back.SchemaVersion != SchemaVersion {
		t.Errorf("schema_version = %q", back.SchemaVersion)
	}
	if back.Hash.Algorithm != Algorithm {
		t.Errorf("hash.algorithm = %q", back.Hash.Algorithm)
	}
	if back.Hash.File == nil || *back.Hash.File != m.ArtifactID {
		t.Errorf("hash.file round-trip failed: %v", back.Hash.File)
	}
	if back.Hash.Tree != nil {
		t.Errorf("hash.tree should be nil, got %v", *back.Hash.Tree)
	}
	if back.Operation == nil || back.Operation.Name != "shape_x" {
		t.Errorf("operation round-trip failed: %+v", back.Operation)
	}
	if len(back.Inputs) != 1 || back.Inputs[0] != m.Inputs[0] {
		t.Errorf("inputs round-trip failed: %v", back.Inputs)
	}
	if back.Table.Rows == nil || *back.Table.Rows != 42 {
		t.Errorf("table.rows round-trip failed: %v", back.Table.Rows)
	}
}

func TestNewArtifactManifestDefaults(t *testing.T) {
	m := NewArtifactManifest("b3:abc", "p", "application/octet-stream")
	if m.SchemaVersion != SchemaVersion {
		t.Errorf("schema_version = %q", m.SchemaVersion)
	}
	if m.Hash.Algorithm != Algorithm {
		t.Errorf("hash.algorithm = %q", m.Hash.Algorithm)
	}
	if m.Inputs == nil {
		t.Fatal("inputs must be non-nil so it marshals as []")
	}
	data, err := m.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	s := string(data)
	if !strings.Contains(s, `"inputs": []`) {
		t.Errorf("empty inputs should marshal as []:\n%s", s)
	}
	if !strings.Contains(s, `"operation": null`) {
		t.Errorf("absent operation should marshal as null:\n%s", s)
	}
	if strings.HasSuffix(s, "\n") {
		t.Error("marshal output must not have a trailing newline (pydantic parity)")
	}
}
