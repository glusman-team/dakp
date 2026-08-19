// Package nercache is the persistent NER mention cache: a Pebble key/value store
// served over localhost HTTP so the Python assertion shapers never re-mine text whose
// (model, config, text) triple has been mined before.
//
// Keys are opaque 64-char lowercase hex BLAKE3 digests produced by the Python client
// (dakp_pipeline.ner.mention_cache); the server never interprets them. Values are
// opaque JSON blobs (a list of serialized mentions), stored and returned verbatim as
// raw JSON bytes so a cache hit round-trips byte-identically.
//
// The store is single-owner: Pebble takes an exclusive lock on the DB directory, so a
// second server over the same directory fails to open it — the Python side reuses the
// already-running server via the server.json discovery file instead.
package nercache

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sync/atomic"

	"github.com/cockroachdb/pebble/v2"
)

// ServerFileName is the discovery file written next to the DB directory on listen:
// {"pid": ..., "port": ...}. Clients reuse a live server instead of spawning one.
const ServerFileName = "server.json"

// Server serves the mention cache over HTTP. Construct with Open; Close with Close.
type Server struct {
	db      *pebble.DB
	mux     *http.ServeMux
	entries atomic.Int64 // approximate entry count (see /stats)
}

// Open opens (creating if needed) the Pebble store at dir and counts its keys.
// A clear error is returned when another live server holds the directory lock.
func Open(dir string) (*Server, error) {
	db, err := pebble.Open(dir, nil)
	if err != nil {
		return nil, fmt.Errorf("nercache: cannot open %s (locked by a running server?): %w", dir, err)
	}
	s := &Server{db: db}
	s.entries.Store(s.countKeys())
	s.routes()
	return s, nil
}

// Close flushes and closes the underlying store.
func (s *Server) Close() error {
	return s.db.Close()
}

// Handler returns the server's HTTP handler (also used with httptest in tests).
func (s *Server) Handler() http.Handler {
	return s.mux
}

func (s *Server) routes() {
	s.mux = http.NewServeMux()
	s.mux.HandleFunc("GET /health", s.handleHealth)
	s.mux.HandleFunc("GET /stats", s.handleStats)
	s.mux.HandleFunc("POST /batch_get", s.handleBatchGet)
	s.mux.HandleFunc("POST /batch_put", s.handleBatchPut)
}

// countKeys scans keys once at open to seed the approximate entry counter. Pebble
// loads values lazily, so this touches key bytes only.
func (s *Server) countKeys() int64 {
	iter, err := s.db.NewIter(nil)
	if err != nil {
		return 0
	}
	defer func() { _ = iter.Close() }()
	var n int64
	for iter.First(); iter.Valid(); iter.Next() {
		n++
	}
	return n
}

// --- wire types ---------------------------------------------------------------

type healthResponse struct {
	OK bool `json:"ok"`
}

type statsResponse struct {
	Entries int64 `json:"entries"`
}

type batchGetRequest struct {
	Keys []string `json:"keys"`
}

type batchGetResponse struct {
	Hits map[string]json.RawMessage `json:"hits"`
}

type batchPutRequest struct {
	Items map[string]json.RawMessage `json:"items"`
}

// --- handlers -----------------------------------------------------------------

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, err error) {
	writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, healthResponse{OK: true})
}

func (s *Server) handleStats(w http.ResponseWriter, _ *http.Request) {
	// Approximate by design: seeded by a key scan at Open, incremented per batch_put
	// item (overwrites inflate it slightly — good enough for a cache-size readout).
	writeJSON(w, http.StatusOK, statsResponse{Entries: s.entries.Load()})
}

func (s *Server) handleBatchGet(w http.ResponseWriter, r *http.Request) {
	var req batchGetRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, fmt.Errorf("batch_get: decode: %w", err))
		return
	}
	hits := make(map[string]json.RawMessage, len(req.Keys))
	for _, key := range req.Keys {
		value, closer, err := s.db.Get([]byte(key))
		if err == pebble.ErrNotFound {
			continue
		}
		if err != nil {
			writeError(w, fmt.Errorf("batch_get: %w", err))
			return
		}
		// Get's slice aliases DB memory: copy before releasing the closer.
		hits[key] = append(json.RawMessage(nil), value...)
		_ = closer.Close()
	}
	writeJSON(w, http.StatusOK, batchGetResponse{Hits: hits})
}

func (s *Server) handleBatchPut(w http.ResponseWriter, r *http.Request) {
	var req batchPutRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, fmt.Errorf("batch_put: decode: %w", err))
		return
	}
	batch := s.db.NewBatch()
	for key, value := range req.Items {
		if err := batch.Set([]byte(key), value, nil); err != nil {
			_ = batch.Close()
			writeError(w, fmt.Errorf("batch_put: %w", err))
			return
		}
	}
	// Sync commit: a confirmed put survives a power loss, so a later run never
	// re-mines text the cache claimed to have stored.
	if err := batch.Commit(pebble.Sync); err != nil {
		writeError(w, fmt.Errorf("batch_put: commit: %w", err))
		return
	}
	_ = batch.Close()
	s.entries.Add(int64(len(req.Items)))
	writeJSON(w, http.StatusOK, healthResponse{OK: true})
}

// --- server.json discovery file -------------------------------------------------

// ServerFile is the contents of the server.json discovery file.
type ServerFile struct {
	PID  int `json:"pid"`
	Port int `json:"port"`
}

// WriteServerFile writes {"pid": ..., "port": ...} to dir/server.json atomically
// (temp file + rename), so clients never observe a half-written discovery file.
func WriteServerFile(dir string, pid, port int) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("nercache: server.json: %w", err)
	}
	data, err := json.Marshal(ServerFile{PID: pid, Port: port})
	if err != nil {
		return fmt.Errorf("nercache: server.json: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".server.json.*")
	if err != nil {
		return fmt.Errorf("nercache: server.json: %w", err)
	}
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmp.Name())
		return fmt.Errorf("nercache: server.json: %w", err)
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmp.Name())
		return fmt.Errorf("nercache: server.json: %w", err)
	}
	if err := os.Rename(tmp.Name(), filepath.Join(dir, ServerFileName)); err != nil {
		_ = os.Remove(tmp.Name())
		return fmt.Errorf("nercache: server.json: %w", err)
	}
	return nil
}

// RemoveServerFile deletes dir/server.json (clean shutdown); missing is not an error.
func RemoveServerFile(dir string) error {
	if err := os.Remove(filepath.Join(dir, ServerFileName)); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("nercache: server.json: %w", err)
	}
	return nil
}
