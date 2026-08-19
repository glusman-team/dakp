// Command dakp-nercache serves DAKP's persistent NER mention cache over localhost HTTP.
//
// The store is a Pebble key/value DB at <workdir>/cache/ner/ (see internal/nercache). The
// server listens on 127.0.0.1 with an ephemeral port and, once listening, writes
// <workdir>/cache/ner/server.json ({"pid", "port"}) atomically so Python clients
// (dakp_pipeline.ner.mention_cache.MentionCache) can discover and reuse it instead of
// spawning a second server — Pebble's exclusive directory lock makes the store
// single-owner, so a second server over the same workdir exits non-zero with a clear
// message. server.json is removed on clean shutdown (SIGTERM/SIGINT).
//
// Usage:
//
//	dakp-nercache --workdir <dir>     # or: dakp-nercache <dir>
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/glusman-team/dakp/go/internal/nercache"
)

// run is the testable body of main: parse args, open the store, listen, serve until
// SIGTERM/SIGINT, then shut down cleanly. Returns the process exit code.
func run(args []string, stderr io.Writer) int {
	fs := flag.NewFlagSet("dakp-nercache", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workdir := fs.String("workdir", "", "pipeline workdir (the cache lives at <workdir>/cache/ner/)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	resolved := *workdir
	if resolved == "" && fs.NArg() == 1 {
		resolved = fs.Arg(0)
	}
	if resolved == "" || fs.NArg() > 1 {
		fmt.Fprintln(stderr, "usage: dakp-nercache --workdir <dir>")
		return 2
	}

	cacheDir := filepath.Join(resolved, "cache", "ner")
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	server, err := nercache.Open(cacheDir)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		_ = server.Close()
		fmt.Fprintln(stderr, "error: listen:", err)
		return 1
	}
	port := listener.Addr().(*net.TCPAddr).Port
	if err := nercache.WriteServerFile(cacheDir, os.Getpid(), port); err != nil {
		_ = listener.Close()
		_ = server.Close()
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	log.Printf("dakp-nercache: serving %s on 127.0.0.1:%d (pid %d)", cacheDir, port, os.Getpid())

	httpServer := &http.Server{Handler: server.Handler(), ReadHeaderTimeout: 30 * time.Second}
	done := make(chan struct{})
	go func() {
		defer close(done)
		if err := httpServer.Serve(listener); err != nil && err != http.ErrServerClosed {
			log.Printf("dakp-nercache: serve: %v", err)
		}
	}()

	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGTERM, syscall.SIGINT)
	select {
	case sig := <-signals:
		log.Printf("dakp-nercache: received %s; shutting down", sig)
	case <-done:
		// The listener died on its own; still run the clean shutdown path.
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := httpServer.Shutdown(ctx); err != nil {
		log.Printf("dakp-nercache: shutdown: %v", err)
	}
	<-done
	if err := server.Close(); err != nil {
		log.Printf("dakp-nercache: close: %v", err)
	}
	if err := nercache.RemoveServerFile(cacheDir); err != nil {
		log.Printf("dakp-nercache: %v", err)
	}
	return 0
}

func main() {
	os.Exit(run(os.Args[1:], os.Stderr))
}
