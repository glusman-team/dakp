package blake3store

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"testing"

	"github.com/zeebo/blake3"
)

// b3IDRe matches a canonical artifact id: b3: + 64 lowercase hex chars (32-byte digest).
var b3IDRe = regexp.MustCompile(`^b3:[0-9a-f]{64}$`)

// Ground-truth vectors computed with the Python reference (src/dakp_pipeline/io/
// content_hash.py, blake3 1.0.9). These lock both the BLAKE3 library choice (32-byte
// default output) and the cross-language parity of every hashing entry point.
const (
	vecEmpty     = "b3:af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262" // BLAKE3(b"")
	vecABC       = "b3:6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85" // BLAKE3(b"abc")
	vecAlphaNL   = "b3:ac678d92b3d739773d18cd952cfcea443fa4a5a98ffc9554b66795bb22d5532d" // BLAKE3(b"alpha\n")
	vecTreeFix   = "b3:3efcf1d2ac7f501dda31fb970875d3a8a2d59852d09f55cf562af3ba3d029fb6" // hash_tree(testdata/tree)
	vecEmptyFile = "b3:a10eaf69278d14f4474196c037c8d29367aea9ccc69700fbf55f9cf67872c3b4" // tree of one 0-byte "empty.bin"
	sriEmpty     = "sha256-47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU="                 // SRI of empty bytes
)

func TestHashBytesKnownVectors(t *testing.T) {
	cases := []struct {
		name string
		in   []byte
		want string
	}{
		{"empty", nil, vecEmpty},
		{"abc", []byte("abc"), vecABC},
		{"alpha_nl", []byte("alpha\n"), vecAlphaNL},
	}
	for _, c := range cases {
		if got := HashBytes(c.in); got != c.want {
			t.Errorf("%s: HashBytes = %q, want %q", c.name, got, c.want)
		}
	}
}

func TestArtifactIDAndDigestDirname(t *testing.T) {
	if got := ArtifactID("deadbeef"); got != "b3:deadbeef" {
		t.Errorf("ArtifactID bare = %q", got)
	}
	if got := ArtifactID("b3:deadbeef"); got != "b3:deadbeef" {
		t.Errorf("ArtifactID idempotent = %q", got)
	}
	if got := DigestDirname("b3:deadbeef"); got != "deadbeef" {
		t.Errorf("DigestDirname = %q", got)
	}
	if got := DigestDirname("deadbeef"); got != "deadbeef" {
		t.Errorf("DigestDirname bare = %q", got)
	}
}

func TestHashFileMatchesBytesAndFixture(t *testing.T) {
	// Fixture file a.txt is exactly "alpha\n".
	got, err := HashFile(filepath.Join("testdata", "tree", "a.txt"))
	if err != nil {
		t.Fatalf("HashFile: %v", err)
	}
	if got != vecAlphaNL {
		t.Errorf("HashFile(a.txt) = %q, want %q", got, vecAlphaNL)
	}
	if !b3IDRe.MatchString(got) {
		t.Errorf("HashFile result not a b3 id: %q", got)
	}

	// A freshly written temp file with identical bytes hashes identically (determinism).
	dir := t.TempDir()
	p := filepath.Join(dir, "x.bin")
	if err := os.WriteFile(p, []byte("alpha\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got2, err := HashFile(p)
	if err != nil {
		t.Fatal(err)
	}
	if got2 != HashBytes([]byte("alpha\n")) || got2 != got {
		t.Errorf("HashFile temp = %q, want %q", got2, got)
	}
}

func TestHashFileMissing(t *testing.T) {
	if _, err := HashFile(filepath.Join(t.TempDir(), "nope.bin")); err == nil {
		t.Fatal("expected error hashing a missing file")
	}
}

// TestHashTreeFixture is the CRITICAL cross-language parity test: the Go tree hash of
// testdata/tree must equal what Python's content_hash.hash_tree produces for the same
// directory (vecTreeFix, computed with the Python implementation). It also asserts the
// result is deterministic across runs and well-formed.
func TestHashTreeFixture(t *testing.T) {
	root := filepath.Join("testdata", "tree")
	got, err := HashTree(root)
	if err != nil {
		t.Fatalf("HashTree: %v", err)
	}
	if got != vecTreeFix {
		t.Fatalf("PARITY MISMATCH: Go HashTree = %q, Python hash_tree = %q", got, vecTreeFix)
	}
	if !b3IDRe.MatchString(got) {
		t.Errorf("HashTree result not a b3 id: %q", got)
	}
	again, err := HashTree(root)
	if err != nil {
		t.Fatal(err)
	}
	if again != got {
		t.Errorf("HashTree not deterministic: %q vs %q", again, got)
	}
}

func TestHashTreeEmptyDir(t *testing.T) {
	// No files -> hasher never updated -> BLAKE3 of the empty input.
	got, err := HashTree(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if got != vecEmpty {
		t.Errorf("empty-dir HashTree = %q, want %q", got, vecEmpty)
	}
}

func TestHashTreeSingleEmptyFile(t *testing.T) {
	// Exercises the size="0" + empty-content framing path; vector from Python.
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "empty.bin"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := HashTree(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got != vecEmptyFile {
		t.Errorf("single-empty-file HashTree = %q, want %q", got, vecEmptyFile)
	}
}

func TestHashTreeMissingRoot(t *testing.T) {
	if _, err := HashTree(filepath.Join(t.TempDir(), "does-not-exist")); err == nil {
		t.Fatal("expected error for a missing tree root")
	}
}

func TestHashFilesBounded(t *testing.T) {
	dir := t.TempDir()
	var paths []string
	want := map[string]string{}
	for _, name := range []string{"a.txt", "b.txt", "c.txt", "d.txt"} {
		p := filepath.Join(dir, name)
		content := []byte("content of " + name)
		if err := os.WriteFile(p, content, 0o644); err != nil {
			t.Fatal(err)
		}
		paths = append(paths, p)
		want[p] = HashBytes(content)
	}

	for _, limit := range []int{0, 1, 2} {
		got, err := HashFiles(context.Background(), paths, limit)
		if err != nil {
			t.Fatalf("HashFiles(limit=%d): %v", limit, err)
		}
		if len(got) != len(want) {
			t.Fatalf("limit=%d: got %d results, want %d", limit, len(got), len(want))
		}
		for p, w := range want {
			if got[p] != w {
				t.Errorf("limit=%d: %s = %q, want %q", limit, p, got[p], w)
			}
		}
	}
}

func TestHashFilesError(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "ok.txt")
	if err := os.WriteFile(good, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := HashFiles(context.Background(), []string{good, filepath.Join(dir, "missing.txt")}, 2)
	if err == nil {
		t.Fatal("expected error when one path is missing")
	}
}

func TestSHA256SRI(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "empty.bin")
	if err := os.WriteFile(p, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := SHA256SRI(p)
	if err != nil {
		t.Fatal(err)
	}
	if got != sriEmpty {
		t.Errorf("SHA256SRI(empty) = %q, want %q", got, sriEmpty)
	}
	if !regexp.MustCompile(`^sha256-[A-Za-z0-9+/]+=$`).MatchString(got) {
		t.Errorf("SHA256SRI malformed: %q", got)
	}
}

func TestHashFileWithSRIMatchesSeparateHashers(t *testing.T) {
	// The single-pass hasher must agree exactly with HashFile + SHA256SRI.
	dir := t.TempDir()
	p := filepath.Join(dir, "blob.bin")
	payload := []byte("dailymed spl fragment")
	for i := 0; i < 10; i++ {
		payload = append(payload, payload...) // ~21 KiB
	}
	if err := os.WriteFile(p, payload, 0o644); err != nil {
		t.Fatal(err)
	}
	id, sri, err := HashFileWithSRI(p)
	if err != nil {
		t.Fatal(err)
	}
	wantID, err := HashFile(p)
	if err != nil {
		t.Fatal(err)
	}
	wantSRI, err := SHA256SRI(p)
	if err != nil {
		t.Fatal(err)
	}
	if id != wantID || sri != wantSRI {
		t.Errorf("HashFileWithSRI = (%q, %q), want (%q, %q)", id, sri, wantID, wantSRI)
	}

	// Known vectors lock both formats for the empty file.
	empty := filepath.Join(dir, "empty.bin")
	if err := os.WriteFile(empty, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	id, sri, err = HashFileWithSRI(empty)
	if err != nil {
		t.Fatal(err)
	}
	if id != vecEmpty || sri != sriEmpty {
		t.Errorf("HashFileWithSRI(empty) = (%q, %q), want (%q, %q)", id, sri, vecEmpty, sriEmpty)
	}

	if _, _, err := HashFileWithSRI(filepath.Join(dir, "missing.bin")); err == nil {
		t.Error("expected error for a missing file")
	}
}

// sriOfBytes computes the expected sha256-<base64> SRI of content independently of the
// implementation under test.
func sriOfBytes(content []byte) string {
	sum := sha256.Sum256(content)
	return "sha256-" + base64.StdEncoding.EncodeToString(sum[:])
}

func TestHashFilesWithSRIBounded(t *testing.T) {
	dir := t.TempDir()
	var paths []string
	want := map[string][2]string{}
	for _, name := range []string{"a.txt", "b.txt", "c.txt", "d.txt"} {
		p := filepath.Join(dir, name)
		content := []byte("content of " + name)
		if err := os.WriteFile(p, content, 0o644); err != nil {
			t.Fatal(err)
		}
		paths = append(paths, p)
		want[p] = [2]string{HashBytes(content), sriOfBytes(content)}
	}
	// An empty file exercises both empty-digest vectors through the batched path.
	empty := filepath.Join(dir, "empty.bin")
	if err := os.WriteFile(empty, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	paths = append(paths, empty)
	want[empty] = [2]string{vecEmpty, sriEmpty}

	for _, limit := range []int{0, 1, 2} {
		got, err := HashFilesWithSRI(context.Background(), paths, limit)
		if err != nil {
			t.Fatalf("HashFilesWithSRI(limit=%d): %v", limit, err)
		}
		if len(got) != len(want) {
			t.Fatalf("limit=%d: got %d results, want %d", limit, len(got), len(want))
		}
		for p, w := range want {
			if got[p] != w {
				t.Errorf("limit=%d: %s = %q, want %q", limit, p, got[p], w)
			}
			// Cross-check against the single-file single-pass hasher.
			id, sri, err := HashFileWithSRI(p)
			if err != nil {
				t.Fatal(err)
			}
			if got[p] != [2]string{id, sri} {
				t.Errorf("limit=%d: %s = %q, HashFileWithSRI = %q", limit, p, got[p], [2]string{id, sri})
			}
		}
	}
}

func TestHashFilesWithSRIError(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "ok.txt")
	if err := os.WriteFile(good, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := HashFilesWithSRI(context.Background(), []string{good, filepath.Join(dir, "missing.txt")}, 2)
	if err == nil {
		t.Fatal("expected error when one path is missing")
	}
}

// hashTreeSequentialReference is a deliberately plain, fully sequential implementation of
// the tree-hash framing (relPath | 0x00 | size | 0x00 | bytes | 0x00, sorted rel paths),
// used to prove the pipelined read-ahead in hashTreeChunked feeds the hasher the exact
// same byte stream.
func hashTreeSequentialReference(t *testing.T, root string) string {
	t.Helper()
	entries, err := listTreeFiles(root)
	if err != nil {
		t.Fatal(err)
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].rel < entries[j].rel })
	h := blake3.New()
	sep := []byte{0}
	for _, e := range entries {
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(e.rel)))
		if err != nil {
			t.Fatal(err)
		}
		h.Write([]byte(e.rel))
		h.Write(sep)
		h.Write([]byte(strconv.FormatInt(e.size, 10)))
		h.Write(sep)
		h.Write(data)
		h.Write(sep)
	}
	return ArtifactID(hex.EncodeToString(h.Sum(nil)))
}

func TestHashTreeReadAheadMatchesSequential(t *testing.T) {
	// A tree that exercises every path of the pipelined implementation: nested dirs, an
	// empty file, many small files, and one file spanning many chunks.
	dir := t.TempDir()
	for i := 0; i < 40; i++ {
		sub := filepath.Join(dir, fmt.Sprintf("sub%02d", i%4))
		if err := os.MkdirAll(sub, 0o755); err != nil {
			t.Fatal(err)
		}
		content := make([]byte, 1000+i*17)
		for j := range content {
			content[j] = byte(i + j)
		}
		if err := os.WriteFile(filepath.Join(sub, fmt.Sprintf("f%02d.bin", i)), content, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "empty.bin"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	big := make([]byte, 17<<20) // 17 MiB: many chunks through the read-ahead queue
	for i := range big {
		big[i] = byte(i * 7)
	}
	if err := os.WriteFile(filepath.Join(dir, "big.bin"), big, 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := HashTree(dir)
	if err != nil {
		t.Fatalf("HashTree: %v", err)
	}
	if want := hashTreeSequentialReference(t, dir); got != want {
		t.Errorf("pipelined HashTree = %q, sequential reference = %q", got, want)
	}
	// Deterministic across runs (read-ahead scheduling must not affect the digest).
	again, err := HashTree(dir)
	if err != nil {
		t.Fatal(err)
	}
	if again != got {
		t.Errorf("HashTree not deterministic: %q vs %q", again, got)
	}
}

func TestHashTreeReadAheadUnreadableFile(t *testing.T) {
	// An unreadable file must surface an error from the read-ahead worker (delivered
	// through its ordered result slot), not a hang or a wrong digest.
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("ok\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	victim := filepath.Join(dir, "b.bin")
	if err := os.WriteFile(victim, []byte("no read"), 0o000); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(victim, 0o644) }) // let TempDir cleanup remove it
	if _, err := os.ReadFile(victim); err == nil {
		t.Skip("chmod 000 did not block reads (running as root?)")
	}
	if _, err := HashTree(dir); err == nil {
		t.Fatal("expected error for an unreadable file in the tree")
	}
}
