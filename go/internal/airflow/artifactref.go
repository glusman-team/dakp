package airflow

import "encoding/json"

// ArtifactRef is the Go mirror of Python io/contracts.ArtifactRef. Paths are strings (XCom
// carries JSON, not Path objects). Nullable fields (rows / schema_fingerprint / manifest) are
// pointers so absent/null round-trips correctly. The JSON tags match the snake_case keys the
// Python side produces/consumes, so refs flow over XCom unchanged in both directions.
type ArtifactRef struct {
	URI               string  `json:"uri"`
	Blake3            string  `json:"blake3"`
	MediaType         string  `json:"media_type"`
	Rows              *int64  `json:"rows"`
	SchemaFingerprint *string `json:"schema_fingerprint"`
	Manifest          *string `json:"manifest"`
}

// Config is the per-run config the bundle reads from the `dakp_config` Airflow Variable (JSON).
// It mirrors the subset of the Python TaskContext/Profile the extract tasks need: where to write
// interim/manifest artifacts (Workdir) and the run knobs.
type Config struct {
	Workdir        string `json:"workdir"`
	Profile        string `json:"profile"`
	FixtureRoot    string `json:"fixture_root"`
	QuarterLimit   int    `json:"quarter_limit"`
	ReleaseLimit   int    `json:"release_limit"`
	Force          bool   `json:"force"`
	Threads        int    `json:"threads"`
	MemoryBudgetGB int    `json:"memory_budget_gb"`
}

// DecodeArtifactRefs converts an XCom value (as returned by sdk.Client.GetXCom — an `any` holding
// a decoded JSON array of objects) into typed ArtifactRefs. It round-trips through encoding/json
// so the result is independent of whether the transport handed us map[string]any vs map[any]any,
// or float64-vs-int numbers (the re-marshal normalizes integral floats before the typed decode).
func DecodeArtifactRefs(v any) ([]ArtifactRef, error) {
	if v == nil {
		return nil, nil
	}
	data, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var refs []ArtifactRef
	if err := json.Unmarshal(data, &refs); err != nil {
		return nil, err
	}
	return refs, nil
}

// EncodeArtifactRefs renders refs as []map[string]any with the exact snake_case keys of the Python
// ArtifactRef contract, for use as a task's return value (the runtime pushes it as the return_value
// XCom). Maps serialize cleanly through both JSON and msgpack with their keys preserved, so the
// downstream Python task reads a list of dicts regardless of the coordinator's wire encoding.
func EncodeArtifactRefs(refs []ArtifactRef) []map[string]any {
	out := make([]map[string]any, 0, len(refs))
	for _, r := range refs {
		m := map[string]any{
			"uri":                r.URI,
			"blake3":             r.Blake3,
			"media_type":         r.MediaType,
			"rows":               nil,
			"schema_fingerprint": nil,
			"manifest":           nil,
		}
		if r.Rows != nil {
			m["rows"] = *r.Rows
		}
		if r.SchemaFingerprint != nil {
			m["schema_fingerprint"] = *r.SchemaFingerprint
		}
		if r.Manifest != nil {
			m["manifest"] = *r.Manifest
		}
		out = append(out, m)
	}
	return out
}
