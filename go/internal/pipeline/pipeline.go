// Package pipeline holds the shared Go types that mirror the Python source-shaping
// contracts in src/dakp_pipeline/io/contracts.py and src/dakp_pipeline/io/downloads.py:
// ArtifactRef handles, the per-task TaskContext, the canonical source_record_id
// derivation, and filename media-type inference. Per-source extractors (a later
// milestone) build on these shared primitives.
package pipeline

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/glusman-team/dakp/go/internal/blake3store"
)

// ArtifactRef is an immutable handle to one pipeline artifact (raw, interim, or tabular).
// It mirrors contracts.ArtifactRef: URI is the concrete path to read from; Blake3 is its
// canonical b3:<hex> content id; Manifest optionally points at the JSON manifest
// describing provenance/inputs.
type ArtifactRef struct {
	URI               string
	Blake3            string
	MediaType         string
	Rows              *int64
	SchemaFingerprint *string
	Manifest          *string
}

// TaskContext is the per-task execution context passed to every fetcher/extractor/
// transformer. It mirrors contracts.TaskContext.
type TaskContext struct {
	Profile        string // mock | sample | wenceslaus_full
	Workdir        string
	FixtureRoot    *string
	Threads        int
	MemoryBudgetGB int
	Params         map[string]any
}

// Fixture returns an ArtifactRef for a fixture file under FixtureRoot, mirroring
// TaskContext.fixture in contracts.py. The ref points directly at the fixture file (it is
// not copied into the content-addressed store).
func (c *TaskContext) Fixture(name string) (ArtifactRef, error) {
	if c.FixtureRoot == nil {
		return ArtifactRef{}, fmt.Errorf("TaskContext.FixtureRoot is nil; cannot resolve fixture")
	}
	path := filepath.Join(*c.FixtureRoot, name)
	id, err := blake3store.HashFile(path)
	if err != nil {
		return ArtifactRef{}, fmt.Errorf("fixture not found: %w", err)
	}
	return ArtifactRef{URI: path, Blake3: id, MediaType: InferMediaType(path)}, nil
}

// SourceRecordID derives a stable per-record b3:<hex> id = BLAKE3 over the \x1f-joined
// (source artifact id, kind, source-local keys). It mirrors spl_xml._source_record_id
// exactly, so Go and Python produce identical ids for the same inputs; joins across
// normalized tables stay stable and re-extraction stays idempotent. Source-specific
// derivations (e.g. FAERS/MEDI/Drugs@FDA string forms) are built on blake3store.HashBytes
// by the per-source extractors.
func SourceRecordID(sourceArtifactID, kind string, localKeys ...string) string {
	parts := make([]string, 0, 2+len(localKeys))
	parts = append(parts, sourceArtifactID, kind)
	parts = append(parts, localKeys...)
	return blake3store.HashBytes([]byte(strings.Join(parts, "\x1f")))
}

// mediaTypes maps filename suffix -> IANA-ish media type (mirrors downloads._MEDIA_TYPES).
var mediaTypes = map[string]string{
	".xml":     "application/xml",
	".xml.gz":  "application/gzip",
	".gz":      "application/gzip",
	".zip":     "application/zip",
	".parquet": "application/vnd.apache.parquet",
	".tsv":     "text/tab-separated-values",
	".csv":     "text/csv",
	".json":    "application/json",
	".jsonl":   "application/x-ndjson",
	".txt":     "text/plain",
	".xlsx":    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

// suffixOrder is the compound-suffix-aware match order (mirrors downloads.infer_media_type):
// longer/compound suffixes are tried before their shorter overlaps (e.g. .xml.gz before
// .gz, .jsonl before .json).
var suffixOrder = []string{".xml.gz", ".tsv", ".csv", ".jsonl", ".json", ".parquet", ".zip", ".gz", ".xml", ".txt", ".xlsx"}

// InferMediaType returns a best-effort media type from a filename, handling compound
// suffixes (e.g. ".xml.gz"). Mirrors downloads.infer_media_type; unknown suffixes yield
// application/octet-stream.
func InferMediaType(path string) string {
	name := strings.ToLower(filepath.Base(path))
	for _, suffix := range suffixOrder {
		if strings.HasSuffix(name, suffix) {
			return mediaTypes[suffix]
		}
	}
	return "application/octet-stream"
}
