package airflow

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestDecodeArtifactRefsNil(t *testing.T) {
	refs, err := DecodeArtifactRefs(nil)
	if err != nil {
		t.Fatalf("DecodeArtifactRefs(nil): %v", err)
	}
	if refs != nil {
		t.Errorf("got %v, want nil", refs)
	}
}

func TestDecodeArtifactRefsInlineList(t *testing.T) {
	rows := int64(7)
	payload := []any{
		map[string]any{"uri": "a.xml.gz", "blake3": "b3:a", "media_type": "application/gzip", "rows": 7, "schema_fingerprint": nil, "manifest": "m/a.json"},
		map[string]any{"uri": "b.xml", "blake3": "b3:b", "media_type": "application/xml", "rows": nil, "schema_fingerprint": nil, "manifest": nil},
	}
	refs, err := DecodeArtifactRefs(payload)
	if err != nil {
		t.Fatalf("DecodeArtifactRefs: %v", err)
	}
	if len(refs) != 2 {
		t.Fatalf("got %d refs, want 2", len(refs))
	}
	if refs[0].URI != "a.xml.gz" || refs[0].Blake3 != "b3:a" || refs[0].Rows == nil || *refs[0].Rows != rows {
		t.Errorf("ref[0] = %+v", refs[0])
	}
	if refs[1].URI != "b.xml" || refs[1].Rows != nil || refs[1].Manifest != nil {
		t.Errorf("ref[1] = %+v", refs[1])
	}
}

// TestDecodeArtifactRefsRefsFile is the single-file handoff: a one-element sentinel payload is
// resolved by reading the store JSON it points at, yielding the exact refs the Python producer
// (refs_to_xcom -> write_refs_manifest) wrote there.
func TestDecodeArtifactRefsRefsFile(t *testing.T) {
	members := []map[string]any{
		{"uri": "data/raw/by-hash/aa/m1.xml", "blake3": "b3:1", "media_type": "application/xml", "rows": nil, "schema_fingerprint": nil, "manifest": "data/manifests/aa.json"},
		{"uri": "data/raw/by-hash/bb/m2.xml.gz", "blake3": "b3:2", "media_type": "application/gzip", "rows": nil, "schema_fingerprint": nil, "manifest": "data/manifests/bb.json"},
	}
	data, err := json.Marshal(members)
	if err != nil {
		t.Fatal(err)
	}
	refsPath := filepath.Join(t.TempDir(), "spl-refs.json")
	if err := os.WriteFile(refsPath, data, 0o644); err != nil {
		t.Fatal(err)
	}
	sentinel := []any{
		map[string]any{"uri": refsPath, "blake3": "b3:refs", "media_type": RefsFileMediaType, "rows": nil, "schema_fingerprint": nil, "manifest": nil},
	}
	refs, err := DecodeArtifactRefs(sentinel)
	if err != nil {
		t.Fatalf("DecodeArtifactRefs(sentinel): %v", err)
	}
	if len(refs) != len(members) {
		t.Fatalf("got %d refs, want %d", len(refs), len(members))
	}
	for i, want := range members {
		if refs[i].URI != want["uri"] || refs[i].Blake3 != want["blake3"] || refs[i].MediaType != want["media_type"] {
			t.Errorf("ref[%d] = %+v, want uri/blake3/media_type of %v", i, refs[i], want)
		}
		if refs[i].Manifest == nil || *refs[i].Manifest != want["manifest"] {
			t.Errorf("ref[%d].Manifest = %v, want %v", i, refs[i].Manifest, want["manifest"])
		}
	}
}

func TestDecodeArtifactRefsRefsFileMissing(t *testing.T) {
	sentinel := []any{
		map[string]any{"uri": filepath.Join(t.TempDir(), "absent.json"), "blake3": "b3:refs", "media_type": RefsFileMediaType},
	}
	if _, err := DecodeArtifactRefs(sentinel); err == nil {
		t.Fatal("expected an error for a missing refs file")
	}
}

func TestDecodeArtifactRefsRefsFileInvalidJSON(t *testing.T) {
	refsPath := filepath.Join(t.TempDir(), "spl-refs.json")
	if err := os.WriteFile(refsPath, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	sentinel := []any{
		map[string]any{"uri": refsPath, "blake3": "b3:refs", "media_type": RefsFileMediaType},
	}
	if _, err := DecodeArtifactRefs(sentinel); err == nil {
		t.Fatal("expected an error for an unreadable refs file")
	}
}

// A genuine one-ref inline list (e.g. the Drugs@FDA acquisition's single ZIP) must NOT be
// mistaken for the handoff sentinel.
func TestDecodeArtifactRefsSingleNonSentinelStaysInline(t *testing.T) {
	payload := []any{
		map[string]any{"uri": "drugsfda.zip", "blake3": "b3:z", "media_type": "application/zip"},
	}
	refs, err := DecodeArtifactRefs(payload)
	if err != nil {
		t.Fatalf("DecodeArtifactRefs: %v", err)
	}
	if len(refs) != 1 || refs[0].URI != "drugsfda.zip" {
		t.Errorf("got %+v, want the single inline ref", refs)
	}
}
