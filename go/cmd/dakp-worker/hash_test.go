package main

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/glusman-team/dakp/go/internal/blake3store"
	"github.com/glusman-team/dakp/go/internal/registry"
)

var b3IDRe = regexp.MustCompile(`^b3:[0-9a-f]{64}$`)

// TestHashSelfRegistered proves the init() self-registration pattern end to end: the
// "hash" command is present in the registry without main.go (or any other file) naming it.
func TestHashSelfRegistered(t *testing.T) {
	if _, ok := registry.Lookup("hash"); !ok {
		t.Fatalf("hash subcommand did not self-register; registered: %v", registry.Names())
	}
}

func TestRunHashFile(t *testing.T) {
	p := filepath.Join(t.TempDir(), "f.txt")
	if err := os.WriteFile(p, []byte("alpha\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := runHash([]string{p}, &buf); err != nil {
		t.Fatalf("runHash: %v", err)
	}
	got := strings.TrimSpace(buf.String())
	if !b3IDRe.MatchString(got) {
		t.Fatalf("not a b3 id: %q", got)
	}
	if want, _ := blake3store.HashFile(p); got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestRunHashTree(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("alpha\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "sub", "b.txt"), []byte("bravo\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := runHash([]string{dir}, &buf); err != nil {
		t.Fatalf("runHash: %v", err)
	}
	got := strings.TrimSpace(buf.String())
	want, err := blake3store.HashTree(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestRunHashModeFlag(t *testing.T) {
	p := filepath.Join(t.TempDir(), "f.txt")
	if err := os.WriteFile(p, []byte("alpha\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// -mode=file on a file matches auto behavior.
	var buf bytes.Buffer
	if err := runHash([]string{"-mode=file", p}, &buf); err != nil {
		t.Fatalf("runHash -mode=file: %v", err)
	}
	if want, _ := blake3store.HashFile(p); strings.TrimSpace(buf.String()) != want {
		t.Fatalf("-mode=file got %q, want %q", buf.String(), want)
	}
}

func TestRunHashErrors(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "f.txt")
	if err := os.WriteFile(good, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name string
		args []string
	}{
		{"no_args", nil},
		{"too_many_args", []string{good, good}},
		{"invalid_mode", []string{"-mode=bogus", good}},
		{"missing_path", []string{filepath.Join(dir, "nope")}},
	}
	for _, c := range cases {
		var buf bytes.Buffer
		if err := runHash(c.args, &buf); err == nil {
			t.Errorf("%s: expected error, got nil (out=%q)", c.name, buf.String())
		}
	}
}

// captureStdout swaps os.Stdout for a pipe while fn runs and returns what was written.
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	defer func() { os.Stdout = old }()

	done := make(chan string, 1)
	go func() {
		var buf bytes.Buffer
		io.Copy(&buf, r)
		done <- buf.String()
	}()
	fn()
	w.Close()
	return <-done
}

// TestHashEndToEndViaRegistry exercises the full path a real invocation takes: init()
// registered "hash" -> registry.Dispatch routes os.Args-style args to it -> it hashes the
// path and prints the id to stdout.
func TestHashEndToEndViaRegistry(t *testing.T) {
	p := filepath.Join(t.TempDir(), "f.txt")
	if err := os.WriteFile(p, []byte("alpha\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	want, err := blake3store.HashFile(p)
	if err != nil {
		t.Fatal(err)
	}

	out := captureStdout(t, func() {
		if err := registry.Dispatch(context.Background(), []string{"dakp-worker", "hash", p}); err != nil {
			t.Errorf("dispatch: %v", err)
		}
	})
	if got := strings.TrimSpace(out); got != want {
		t.Fatalf("end-to-end hash got %q, want %q", got, want)
	}
}
