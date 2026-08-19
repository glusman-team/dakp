package nercache

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

// hexKey returns a deterministic 64-char hex-looking key for tests.
func hexKey(c byte) string {
	return string(bytes.Repeat([]byte{c}, 64))
}

func put(t *testing.T, base string, items map[string]json.RawMessage) {
	t.Helper()
	body, err := json.Marshal(batchPutRequest{Items: items})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(base+"/batch_put", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("batch_put status = %d", resp.StatusCode)
	}
}

func get(t *testing.T, base string, keys ...string) batchGetResponse {
	t.Helper()
	body, err := json.Marshal(batchGetRequest{Keys: keys})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(base+"/batch_get", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("batch_get status = %d", resp.StatusCode)
	}
	var out batchGetResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestBatchPutGetRoundTrip(t *testing.T) {
	s, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s.Close() }()
	httpServer := httptest.NewServer(s.Handler())
	defer httpServer.Close()

	// Health + empty stats.
	resp, err := http.Get(httpServer.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("health status = %d", resp.StatusCode)
	}

	keyA, keyB, keyMiss := hexKey('a'), hexKey('b'), hexKey('f')
	valueA := json.RawMessage(`[{"text":"asthma","start":0,"end":6}]`)
	put(t, httpServer.URL, map[string]json.RawMessage{keyA: valueA, keyB: json.RawMessage(`[]`)})

	got := get(t, httpServer.URL, keyA, keyB, keyMiss)
	if len(got.Hits) != 2 {
		t.Fatalf("hits = %d, want 2 (miss must be absent)", len(got.Hits))
	}
	if !bytes.Equal(got.Hits[keyA], valueA) {
		t.Fatalf("value round-trip mismatch: got %s want %s", got.Hits[keyA], valueA)
	}
	if string(got.Hits[keyB]) != `[]` {
		t.Fatalf("empty-list value mismatch: got %s", got.Hits[keyB])
	}

	// Stats reflects the puts (approximate contract).
	resp, err = http.Get(httpServer.URL + "/stats")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()
	var stats statsResponse
	if err := json.NewDecoder(resp.Body).Decode(&stats); err != nil {
		t.Fatal(err)
	}
	if stats.Entries != 2 {
		t.Fatalf("entries = %d, want 2", stats.Entries)
	}
}

func TestPersistenceAcrossRestart(t *testing.T) {
	dir := t.TempDir()

	s, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	first := httptest.NewServer(s.Handler())
	key := hexKey('c')
	value := json.RawMessage(`[{"text":"hypertension"}]`)
	put(t, first.URL, map[string]json.RawMessage{key: value})
	first.Close()
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}

	// Reopen the same directory: the put must survive.
	reopened, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = reopened.Close() }()
	second := httptest.NewServer(reopened.Handler())
	defer second.Close()
	got := get(t, second.URL, key)
	if !bytes.Equal(got.Hits[key], value) {
		t.Fatalf("after restart: got %s want %s", got.Hits[key], value)
	}
	if reopened.entries.Load() != 1 {
		t.Fatalf("entries after reopen = %d, want 1 (key scan at Open)", reopened.entries.Load())
	}
}

func TestServerFileAtomicWriteAndRemove(t *testing.T) {
	dir := t.TempDir()
	if err := WriteServerFile(dir, 1234, 5678); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(dir, ServerFileName))
	if err != nil {
		t.Fatal(err)
	}
	var sf ServerFile
	if err := json.Unmarshal(data, &sf); err != nil {
		t.Fatal(err)
	}
	if sf.PID != 1234 || sf.Port != 5678 {
		t.Fatalf("server.json = %+v", sf)
	}
	// No temp files left behind.
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Fatalf("dir holds %d entries, want exactly server.json", len(entries))
	}
	if err := RemoveServerFile(dir); err != nil {
		t.Fatal(err)
	}
	if err := RemoveServerFile(dir); err != nil {
		t.Fatalf("removing a missing server.json must not error: %v", err)
	}
}
