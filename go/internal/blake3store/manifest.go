package blake3store

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// SchemaVersion is the artifact-manifest schema version (matches manifests.py).
const SchemaVersion = "dakp.artifact.v1"

// HashBlock records the content hashes for one artifact. Algorithm is always "BLAKE3";
// File / Tree / SHA256SRI are nullable (nil -> JSON null), mirroring the Python model.
type HashBlock struct {
	Algorithm string  `json:"algorithm"`
	File      *string `json:"file"`
	Tree      *string `json:"tree"`
	SHA256SRI *string `json:"sha256_sri"`
}

// NewHashBlock returns a HashBlock with Algorithm defaulted to BLAKE3, mirroring the
// Python HashBlock default.
func NewHashBlock() HashBlock { return HashBlock{Algorithm: Algorithm} }

// OperationBlock identifies the operation that produced an artifact.
type OperationBlock struct {
	Name       string  `json:"name"`
	Version    string  `json:"version"`
	ConfigHash *string `json:"config_hash"`
}

// SourceBlock records where a raw artifact came from.
type SourceBlock struct {
	URL          *string `json:"url"`
	ETag         *string `json:"etag"`
	LastModified *string `json:"last_modified"`
	RetrievedAt  *string `json:"retrieved_at"`
}

// EnvironmentBlock records the build environment provenance.
type EnvironmentBlock struct {
	GitCommit        *string `json:"git_commit"`
	UVLockHash       *string `json:"uv_lock_hash"`
	TablassertCommit *string `json:"tablassert_commit"`
	FullmapHash      *string `json:"fullmap_hash"`
}

// TableBlock records tabular-artifact statistics.
type TableBlock struct {
	Rows              *int64  `json:"rows"`
	Partitions        *int64  `json:"partitions"`
	SchemaFingerprint *string `json:"schema_fingerprint"`
	Warnings          *int64  `json:"warnings"`
}

// ArtifactManifest is the provenance record for one content-addressed artifact. Field
// order matches the Python pydantic model (manifests.ArtifactManifest) so Go and Python
// JSON agree field-by-field; see Marshal for the byte-parity encoding rules.
type ArtifactManifest struct {
	SchemaVersion string           `json:"schema_version"`
	ArtifactID    string           `json:"artifact_id"`
	Path          string           `json:"path"`
	MediaType     string           `json:"media_type"`
	Hash          HashBlock        `json:"hash"`
	Inputs        []string         `json:"inputs"`
	Operation     *OperationBlock  `json:"operation"`
	Source        SourceBlock      `json:"source"`
	Environment   EnvironmentBlock `json:"environment"`
	Table         TableBlock       `json:"table"`
}

// NewArtifactManifest returns a manifest with schema_version defaulted and the always-
// present blocks (hash/source/environment/table) initialized to their zero/default values,
// mirroring the Python ArtifactManifest defaults. Inputs defaults to an empty (non-nil)
// slice so it marshals as [] rather than null.
func NewArtifactManifest(artifactID, path, mediaType string) *ArtifactManifest {
	return &ArtifactManifest{
		SchemaVersion: SchemaVersion,
		ArtifactID:    artifactID,
		Path:          path,
		MediaType:     mediaType,
		Hash:          NewHashBlock(),
		Inputs:        []string{},
	}
}

// StringPtr returns a pointer to v, for populating nullable manifest fields.
func StringPtr(v string) *string { return &v }

// IntPtr returns a pointer to v, for populating nullable manifest fields.
func IntPtr(v int64) *int64 { return &v }

// Marshal renders the manifest with the same 2-space indentation and field order as
// Python's pydantic model_dump_json(indent=2): HTML escaping disabled and no trailing
// newline, so Go and Python produce byte-identical manifests for the same data (see
// manifest_test.go, which asserts this against Python-generated golden fixtures).
func (m *ArtifactManifest) Marshal() ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(m); err != nil {
		return nil, err
	}
	// json.Encoder appends a trailing newline; pydantic does not. Trim it for byte parity.
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// WriteManifest writes m to path as indented JSON (creating parent dirs), matching the
// Python ArtifactManifest.write shape.
func WriteManifest(path string, m *ArtifactManifest) error {
	data, err := m.Marshal()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// ReadManifest reads and normalizes a manifest JSON file written by Python or Go. A null
// or absent inputs array is normalized to an empty slice so re-marshalling matches the
// Python default ([]).
func ReadManifest(path string) (*ArtifactManifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m ArtifactManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("parse manifest %s: %w", path, err)
	}
	if m.Inputs == nil {
		m.Inputs = []string{}
	}
	return &m, nil
}
