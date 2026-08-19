// Package blake3store implements DAKP's BLAKE3 content addressing: streaming file
// hashes, a deterministic Nix-NAR-like directory/tree hash, the optional SHA-256 SRI
// interoperability hash, and the artifact-manifest provenance record (see manifest.go).
//
// It is the Go mirror of the Python reference implementation in
// src/dakp_pipeline/io/content_hash.py and src/dakp_pipeline/io/manifests.py. The tree
// hash is byte-for-byte compatible with Python: the same directory yields the same
// b3:<hex> id whether hashed by Python or Go (see HashTree for the exact algorithm and
// blake3store_test.go for the cross-language parity fixtures).
package blake3store

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/zeebo/blake3"
	"golang.org/x/sync/errgroup"
)

// Algorithm is the primary DAKP content-hash algorithm recorded in manifests.
const Algorithm = "BLAKE3"

// IDPrefix prefixes every canonical artifact id.
const IDPrefix = "b3:"

// DefaultChunk is the streaming read window (1 MiB), matching the Python reference.
const DefaultChunk = 1 << 20

// ArtifactID normalizes a bare hex digest into the canonical b3:<hex> artifact id. It is
// idempotent: an id that already carries the prefix is returned unchanged.
func ArtifactID(hexDigest string) string {
	if strings.HasPrefix(hexDigest, IDPrefix) {
		return hexDigest
	}
	return IDPrefix + hexDigest
}

// DigestDirname strips the b3: prefix, returning the bare hex digest used as the
// content-addressed store directory name.
func DigestDirname(id string) string {
	return strings.TrimPrefix(id, IDPrefix)
}

// HashBytes returns the BLAKE3 of data as a b3:<hex> id (32-byte / 64-hex digest).
func HashBytes(data []byte) string {
	sum := blake3.Sum256(data)
	return ArtifactID(hex.EncodeToString(sum[:]))
}

// HashFile streams a file's bytes through BLAKE3 (1 MiB window) and returns b3:<hex>.
func HashFile(path string) (string, error) {
	return hashFileChunked(path, DefaultChunk)
}

func hashFileChunked(path string, chunk int) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := blake3.New()
	if _, err := io.CopyBuffer(h, f, make([]byte, chunk)); err != nil {
		return "", err
	}
	return ArtifactID(hex.EncodeToString(h.Sum(nil))), nil
}

// treeEntry is one regular file discovered under a tree root, with the size captured
// during the walk so hashing does not re-stat.
type treeEntry struct {
	rel  string // relative POSIX path (forward slashes)
	size int64
}

// HashTree computes a deterministic BLAKE3 tree hash over a directory, mirroring the
// Python reference (content_hash.hash_tree) EXACTLY so Python and Go agree byte-for-byte.
// The algorithm:
//
//  1. collect every regular file under root (recursive);
//  2. sort by relative POSIX path (forward slashes, lexicographic by byte value, which
//     equals Unicode code-point order for valid UTF-8 — the same order Python uses);
//  3. feed ONE BLAKE3 hasher, per file in order:
//     relPath(utf-8) | 0x00 | size(decimal ascii) | 0x00 | fileBytes | 0x00
//  4. return b3:<hex> of the final 32-byte digest.
//
// Directory mtimes, traversal order, and empty directories do not affect the result; an
// empty directory hashes to BLAKE3 of the empty input. Symlinks and other non-regular
// files are skipped (the Python reference follows symlinked files to their target
// content; DAKP fixtures and production trees use regular files).
func HashTree(root string) (string, error) {
	return hashTreeChunked(root, DefaultChunk)
}

func hashTreeChunked(root string, chunk int) (string, error) {
	entries, err := listTreeFiles(root)
	if err != nil {
		return "", err
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].rel < entries[j].rel })

	h := blake3.New()
	buf := make([]byte, chunk)
	sep := []byte{0}
	for _, e := range entries {
		abs := filepath.Join(root, filepath.FromSlash(e.rel))
		// hash.Hash.Write never returns an error (interface contract), so the framing
		// writes are unchecked; file content is streamed via streamFile, which can fail.
		h.Write([]byte(e.rel))
		h.Write(sep)
		h.Write([]byte(strconv.FormatInt(e.size, 10)))
		h.Write(sep)
		if err := streamFile(abs, h, buf); err != nil {
			return "", err
		}
		h.Write(sep)
	}
	return ArtifactID(hex.EncodeToString(h.Sum(nil))), nil
}

// listTreeFiles walks root and returns every regular file with its relative POSIX path
// and size. Directories are recursed into; symlinks and other non-regular files are
// skipped. A missing root yields an error.
func listTreeFiles(root string) ([]treeEntry, error) {
	var entries []treeEntry
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if !d.Type().IsRegular() {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		entries = append(entries, treeEntry{rel: filepath.ToSlash(rel), size: info.Size()})
		return nil
	})
	if err != nil {
		return nil, err
	}
	return entries, nil
}

// streamFile copies path's bytes into w using buf as scratch space.
func streamFile(path string, w io.Writer, buf []byte) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.CopyBuffer(w, f, buf)
	return err
}

// HashFiles hashes many files concurrently with bounded parallelism, returning a map from
// input path to b3:<hex> file id. It uses errgroup with SetLimit so callers can respect
// Airflow task concurrency / memory budgets, and it cancels remaining work on the first
// error (cooperatively, at file granularity). limit <= 0 means unbounded.
func HashFiles(ctx context.Context, paths []string, limit int) (map[string]string, error) {
	var (
		mu  sync.Mutex
		out = make(map[string]string, len(paths))
	)
	g, gctx := errgroup.WithContext(ctx)
	if limit > 0 {
		g.SetLimit(limit)
	}
	for _, p := range paths {
		p := p
		g.Go(func() error {
			if err := gctx.Err(); err != nil {
				return err // ctx cancelled or another goroutine already failed
			}
			id, err := HashFile(p)
			if err != nil {
				return fmt.Errorf("hash %s: %w", p, err)
			}
			mu.Lock()
			out[p] = id
			mu.Unlock()
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}
	return out, nil
}

// SHA256SRI computes the optional secondary Subresource Integrity string (sha256-<base64>)
// for a file, mirroring content_hash.sha256_sri. It is interoperability sugar only; the
// canonical DAKP id is always BLAKE3.
func SHA256SRI(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return "sha256-" + base64.StdEncoding.EncodeToString(h.Sum(nil)), nil
}

// HashFileWithSRI streams a file's bytes ONCE, feeding both the BLAKE3 and SHA-256 hashers
// per chunk, and returns (b3:<hex>, sha256-<base64>) — the same formats as HashFile and
// SHA256SRI (the Go mirror of content_hash.hash_file_with_sri).
func HashFileWithSRI(path string) (string, string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", "", err
	}
	defer f.Close()
	b3 := blake3.New()
	sha := sha256.New()
	if _, err := io.CopyBuffer(io.MultiWriter(b3, sha), f, make([]byte, DefaultChunk)); err != nil {
		return "", "", err
	}
	return ArtifactID(hex.EncodeToString(b3.Sum(nil))), "sha256-" + base64.StdEncoding.EncodeToString(sha.Sum(nil)), nil
}
