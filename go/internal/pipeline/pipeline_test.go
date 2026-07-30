package pipeline

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/glusman-team/dakp/go/internal/blake3store"
)

var b3IDRe = regexp.MustCompile(`^b3:[0-9a-f]{64}$`)

// TestSourceRecordIDMatchesPython locks parity with spl_xml._source_record_id. Vectors
// were computed with the Python implementation (hash_bytes over the \x1f-joined parts).
func TestSourceRecordIDMatchesPython(t *testing.T) {
	srcAB := "b3:" + strings.Repeat("ab", 32)
	src00 := "b3:" + strings.Repeat("00", 32)
	cases := []struct {
		name string
		src  string
		kind string
		keys []string
		want string
	}{
		{"section", srcAB, "section", []string{"set1", "34067-9"},
			"b3:1c31ed19f48e44c8f3ea85b64a15734026e34a4a855c28787aeeb44d0c1f39e4"},
		{"set", src00, "set", []string{"SOMESET"},
			"b3:649f830dd99f0ac0f18176db665fab924dd0e18c3462be5c015d73cdf8d1d4a8"},
		{"no_local_keys", src00, "set", nil,
			"b3:9192432b470b7a82adac56fa536c5a807bb650ceca213d64db0fd1cb8e234f23"},
	}
	for _, c := range cases {
		if got := SourceRecordID(c.src, c.kind, c.keys...); got != c.want {
			t.Errorf("%s: SourceRecordID = %q, want %q", c.name, got, c.want)
		}
	}
}

func TestSourceRecordIDDeterministicAndDistinct(t *testing.T) {
	src := "b3:" + strings.Repeat("cd", 32)
	a := SourceRecordID(src, "set", "K1")
	if a != SourceRecordID(src, "set", "K1") {
		t.Error("SourceRecordID is not deterministic")
	}
	if !b3IDRe.MatchString(a) {
		t.Errorf("not a b3 id: %q", a)
	}
	if a == SourceRecordID(src, "section", "K1") {
		t.Error("different kind must yield a different id")
	}
	if a == SourceRecordID(src, "set", "K2") {
		t.Error("different local key must yield a different id")
	}
}

func TestInferMediaType(t *testing.T) {
	cases := []struct {
		path string
		want string
	}{
		{"x.xml", "application/xml"},
		{"x.xml.gz", "application/gzip"}, // compound suffix wins over .gz
		{"x.gz", "application/gzip"},
		{"x.zip", "application/zip"},
		{"x.parquet", "application/vnd.apache.parquet"},
		{"x.tsv", "text/tab-separated-values"},
		{"x.csv", "text/csv"},
		{"x.jsonl", "application/x-ndjson"}, // .jsonl wins over .json
		{"x.json", "application/json"},
		{"x.txt", "text/plain"},
		{"x.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
		{"X.XML", "application/xml"}, // case-insensitive
		{"x.unknown", "application/octet-stream"},
		{"noext", "application/octet-stream"},
		{"/some/dir/x.tsv", "text/tab-separated-values"}, // only the base name matters
	}
	for _, c := range cases {
		if got := InferMediaType(c.path); got != c.want {
			t.Errorf("InferMediaType(%q) = %q, want %q", c.path, got, c.want)
		}
	}
}

func TestFixture(t *testing.T) {
	root := t.TempDir()
	content := []byte("fixture bytes\n")
	if err := os.WriteFile(filepath.Join(root, "sample.tsv"), content, 0o644); err != nil {
		t.Fatal(err)
	}
	ctx := &TaskContext{Profile: "mock", Workdir: t.TempDir(), FixtureRoot: &root}

	ref, err := ctx.Fixture("sample.tsv")
	if err != nil {
		t.Fatalf("Fixture: %v", err)
	}
	if ref.Blake3 != blake3store.HashBytes(content) {
		t.Errorf("Fixture blake3 = %q, want %q", ref.Blake3, blake3store.HashBytes(content))
	}
	if ref.MediaType != "text/tab-separated-values" {
		t.Errorf("Fixture media type = %q", ref.MediaType)
	}
	if ref.URI != filepath.Join(root, "sample.tsv") {
		t.Errorf("Fixture URI = %q", ref.URI)
	}
}

func TestFixtureErrors(t *testing.T) {
	// Nil fixture root.
	if _, err := (&TaskContext{}).Fixture("x.tsv"); err == nil {
		t.Error("expected error for nil FixtureRoot")
	}
	// Missing fixture file.
	root := t.TempDir()
	if _, err := (&TaskContext{FixtureRoot: &root}).Fixture("absent.tsv"); err == nil {
		t.Error("expected error for a missing fixture")
	}
}
