package airflow

import (
	"path/filepath"
	"strings"

	"github.com/glusman-team/dakp/go/internal/blake3store"
)

// ParquetMediaType is the media type for interim parquet tables (matches Python
// schemas.PARQUET_MEDIA_TYPE).
const ParquetMediaType = "application/vnd.apache.parquet"

// SchemaFingerprint computes the deterministic b3:<hex> fingerprint of an ordered column list
// (matches Python schemas.schema_fingerprint: BLAKE3 of the tab-joined columns). Captured in
// artifact manifests so schema drift is detectable by hash.
func SchemaFingerprint(columns []string) string {
	return blake3store.HashBytes([]byte(strings.Join(columns, "\t")))
}

// TSVMediaType is the media type for uncompressed Tablassert-handoff TSV tables (matches Python
// schemas.TSV_MEDIA_TYPE).
const TSVMediaType = "text/tab-separated-values"

// RegisterInput describes one in-place workdir output to register.
type RegisterInput struct {
	Path              string
	MediaType         string
	Rows              int64
	SchemaFingerprint string
	Inputs            []string // provenance chain: the input artifact ids
	Warnings          int64
	Operation         string // operation-block name ("" omits the operation block)
}

// Store is the Go mirror of Python io/artifact_store.ArtifactStore, bound to a workdir root. It
// registers workdir outputs (interim parquet) in place: hash + sidecar manifest, no copy. The
// directory layout mirrors Python paths.Workdir (data/interim, data/manifests, data/raw/by-hash).
type Store struct {
	Workdir string
}

// InterimDir returns the partitioned interim-table root (data/interim).
func (s Store) InterimDir() string { return filepath.Join(s.Workdir, "data", "interim") }

// TabularDir returns the uncompressed Tablassert-handoff TSV root (data/tabular).
func (s Store) TabularDir() string { return filepath.Join(s.Workdir, "data", "tabular") }

// ManifestsDir returns the per-artifact manifest root (data/manifests).
func (s Store) ManifestsDir() string { return filepath.Join(s.Workdir, "data", "manifests") }

// ManifestPath returns the sidecar manifest path for an artifact id (manifests/<hex>.json),
// mirroring ArtifactStore.manifest_path.
func (s Store) ManifestPath(id string) string {
	return filepath.Join(s.ManifestsDir(), blake3store.DigestDirname(id)+".json")
}

// Register mirrors ArtifactStore.register for an in-place workdir output: hashes the file, writes
// the sidecar manifest at manifests/<hex>.json, and returns an ArtifactRef whose URI is the path.
// The manifest's table block carries rows/schema_fingerprint/warnings; the operation block (when
// named) records the producing stage. The returned ref's uri/blake3/media_type/rows/
// schema_fingerprint/manifest are what the downstream shaping stage consumes.
func (s Store) Register(in RegisterInput) (ArtifactRef, error) {
	id, err := blake3store.HashFile(in.Path)
	if err != nil {
		return ArtifactRef{}, err
	}
	sri, err := blake3store.SHA256SRI(in.Path)
	if err != nil {
		return ArtifactRef{}, err
	}
	m := blake3store.NewArtifactManifest(id, in.Path, in.MediaType)
	m.Hash.File = blake3store.StringPtr(id)
	m.Hash.SHA256SRI = blake3store.StringPtr(sri)
	if in.Inputs != nil {
		m.Inputs = in.Inputs
	}
	if in.Operation != "" {
		m.Operation = &blake3store.OperationBlock{Name: in.Operation}
	}
	m.Table.Rows = blake3store.IntPtr(in.Rows)
	m.Table.SchemaFingerprint = blake3store.StringPtr(in.SchemaFingerprint)
	m.Table.Warnings = blake3store.IntPtr(in.Warnings)

	mpath := s.ManifestPath(id)
	if err := blake3store.WriteManifest(mpath, m); err != nil {
		return ArtifactRef{}, err
	}
	return ArtifactRef{
		URI:               in.Path,
		Blake3:            id,
		MediaType:         in.MediaType,
		Rows:              blake3store.IntPtr(in.Rows),
		SchemaFingerprint: blake3store.StringPtr(in.SchemaFingerprint),
		Manifest:          blake3store.StringPtr(mpath),
	}, nil
}
