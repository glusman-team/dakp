package airflow

import (
	"encoding/hex"
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/zeebo/blake3"
	"golang.org/x/sys/unix"

	"github.com/glusman-team/dakp/go/internal/blake3store"
)

// opIndexFilename is the operation-index sidecar inside manifests (matches Python
// artifact_store.OP_INDEX_FILENAME).
const opIndexFilename = "_index.json"

// Operation-index key spec, mirrored BYTE-FOR-BYTE from Python
// src/dakp_pipeline/io/artifact_store.py (op_index_key) — keep the two in lockstep:
//
//   - canonical = operation + "|" + strings.Join(sorted(inputs), "|"); sorting is byte-wise
//     (Go sort.Strings), which equals Python's Unicode code-point order for these ASCII ids;
//   - key = hex(BLAKE3(canonical)) — 64 lowercase hex chars, WITHOUT the "b3:" prefix.
//
// The index JSON is {"version": 1, "entries": {<key>: {"operation", "inputs": [sorted ids],
// "outputs": [{"artifact_id", "path", "media_type", "rows", "schema_fingerprint"}]}}}.
type opIndexFile struct {
	Version int                     `json:"version"`
	Entries map[string]opIndexEntry `json:"entries"`
}

type opIndexEntry struct {
	Operation string          `json:"operation"`
	Inputs    []string        `json:"inputs"`
	Outputs   []opIndexOutput `json:"outputs"`
}

type opIndexOutput struct {
	ArtifactID        string  `json:"artifact_id"`
	Path              string  `json:"path"`
	MediaType         string  `json:"media_type"`
	Rows              *int64  `json:"rows"`
	SchemaFingerprint *string `json:"schema_fingerprint"`
}

// OpIndexKey computes the 64-hex BLAKE3 index key for an operation + its exact input id set,
// mirroring Python op_index_key. Inputs are sorted (on a copy) so call-site ordering never
// fragments the cache.
func OpIndexKey(operation string, inputs []string) string {
	sorted := append([]string(nil), inputs...)
	sort.Strings(sorted)
	sum := blake3.Sum256([]byte(operation + "|" + strings.Join(sorted, "|")))
	return hex.EncodeToString(sum[:])
}

// opIndexPath returns the index sidecar path (manifests/_index.json).
func (s Store) opIndexPath() string { return filepath.Join(s.ManifestsDir(), opIndexFilename) }

// readOpIndex is a best-effort read of the index (writers replace atomically, so reads need
// no lock); a missing/corrupt file reads as an empty index.
func (s Store) readOpIndex() *opIndexFile {
	idx := &opIndexFile{Version: 1, Entries: map[string]opIndexEntry{}}
	data, err := os.ReadFile(s.opIndexPath())
	if err != nil {
		return idx
	}
	if err := json.Unmarshal(data, idx); err != nil || idx.Entries == nil {
		return &opIndexFile{Version: 1, Entries: map[string]opIndexEntry{}}
	}
	return idx
}

// withOpIndexLock serializes a read-modify-write of the index on a kernel flock (the same
// mechanism as the Python side and the per-GPU model locks): registration is sequential
// within a task but concurrent ACROSS the three parallel extract tasks.
func (s Store) withOpIndexLock(fn func(idx *opIndexFile) error) error {
	if err := os.MkdirAll(s.ManifestsDir(), 0o755); err != nil {
		return err
	}
	lock, err := os.OpenFile(filepath.Join(s.ManifestsDir(), "_index.lock"), os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return err
	}
	defer lock.Close()
	if err := unix.Flock(int(lock.Fd()), unix.LOCK_EX); err != nil {
		return err
	}
	defer unix.Flock(int(lock.Fd()), unix.LOCK_UN)
	idx := s.readOpIndex()
	if err := fn(idx); err != nil {
		return err
	}
	data, err := json.MarshalIndent(idx, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(s.ManifestsDir(), "._index.json.*")
	if err != nil {
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmp.Name())
		return err
	}
	return os.Rename(tmp.Name(), s.opIndexPath())
}

// FindByOperation returns the previously registered outputs of operation over exactly
// inputs, or nil — the Go mirror of Python ArtifactStore.find_by_operation backing the
// "already done" skip in the extract tasks. Every referenced output path and manifest must
// still exist and parse; a stale entry is pruned and reported as a miss so the caller simply
// redoes the work.
func (s Store) FindByOperation(operation string, inputs []string) []ArtifactRef {
	if len(inputs) == 0 {
		return nil
	}
	key := OpIndexKey(operation, inputs)
	entry, ok := s.readOpIndex().Entries[key]
	sorted := append([]string(nil), inputs...)
	sort.Strings(sorted)
	if !ok || entry.Operation != operation || !equalStrings(entry.Inputs, sorted) || len(entry.Outputs) == 0 {
		return nil
	}
	refs := make([]ArtifactRef, 0, len(entry.Outputs))
	for _, out := range entry.Outputs {
		if out.ArtifactID == "" || out.Path == "" || out.MediaType == "" {
			return s.pruneOpIndex(key)
		}
		if _, err := os.Stat(out.Path); err != nil {
			return s.pruneOpIndex(key)
		}
		// The manifest must exist and belong to the entry; per-manifest operation/inputs
		// equality is deliberately NOT enforced — each fan-out member is registered under its
		// own per-artifact operation name and parsed-input id list, while the entry records
		// the task-level operation (same rule as the Python lookup).
		manifest, err := blake3store.ReadManifest(s.ManifestPath(out.ArtifactID))
		if err != nil || manifest.ArtifactID != out.ArtifactID {
			return s.pruneOpIndex(key)
		}
		refs = append(refs, ArtifactRef{
			URI:               out.Path,
			Blake3:            out.ArtifactID,
			MediaType:         out.MediaType,
			Rows:              out.Rows,
			SchemaFingerprint: out.SchemaFingerprint,
			Manifest:          blake3store.StringPtr(s.ManifestPath(out.ArtifactID)),
		})
	}
	return refs
}

// RecordOperation atomically replaces the index entry for operation + inputs with refs
// (matching Python ArtifactStore.record_operation). A failed index write is logged, never
// fatal — it only costs the next run its skip.
func (s Store) RecordOperation(operation string, inputs []string, refs []ArtifactRef) {
	outputs := make([]opIndexOutput, 0, len(refs))
	for _, ref := range refs {
		outputs = append(outputs, opIndexOutput{
			ArtifactID:        ref.Blake3,
			Path:              ref.URI,
			MediaType:         ref.MediaType,
			Rows:              ref.Rows,
			SchemaFingerprint: ref.SchemaFingerprint,
		})
	}
	sorted := append([]string(nil), inputs...)
	sort.Strings(sorted)
	err := s.withOpIndexLock(func(idx *opIndexFile) error {
		idx.Entries[OpIndexKey(operation, inputs)] = opIndexEntry{Operation: operation, Inputs: sorted, Outputs: outputs}
		return nil
	})
	if err != nil {
		slog.Warn("op index record failed (skip disabled for " + operation + "): " + err.Error())
	}
}

// pruneOpIndex drops a stale index entry (missing artifact/manifest) and reports a miss.
func (s Store) pruneOpIndex(key string) []ArtifactRef {
	_ = s.withOpIndexLock(func(idx *opIndexFile) error {
		delete(idx.Entries, key)
		return nil
	})
	return nil
}

// upstreamInputIDs collects the blake3 ids of the upstream refs — the "inputs haven't
// changed" key material for the already-done skip (the refs arrived over XCom from
// acquisition, so no re-hashing is needed).
func upstreamInputIDs(inputs []ArtifactRef) []string {
	ids := make([]string, 0, len(inputs))
	for _, ref := range inputs {
		if ref.Blake3 != "" {
			ids = append(ids, ref.Blake3)
		}
	}
	return ids
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
