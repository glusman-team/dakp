package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/glusman-team/dakp/go/internal/blake3store"
	"github.com/glusman-team/dakp/go/internal/registry"
)

// init self-registers the "hash" subcommand. This is the self-registration pattern every
// extractor subcommand follows: a new file in package main with an init() that calls
// registry.Register. No other file (especially main.go) needs to change.
func init() {
	registry.Register("hash", func(_ context.Context, args []string) error {
		return runHash(args, os.Stdout)
	})
}

// runHash implements the `hash` subcommand: BLAKE3-hash a path and print the canonical
// b3:<hex> id to stdout (the only thing written to stdout, so callers can capture it).
// A directory is tree-hashed (blake3store.HashTree); a regular file is content-hashed
// (blake3store.HashFile). -mode forces one or the other ("auto" picks by path type).
func runHash(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("hash", flag.ContinueOnError)
	fs.SetOutput(io.Discard) // keep stdout clean for the id; errors are returned, not printed
	mode := fs.String("mode", "auto", "hash mode: auto | file | tree")
	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("hash: %w", err)
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("hash: expected exactly one path argument, got %d", fs.NArg())
	}
	path := fs.Arg(0)

	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("hash: %w", err)
	}

	useTree := info.IsDir()
	switch *mode {
	case "auto":
	case "tree":
		useTree = true
	case "file":
		useTree = false
	default:
		return fmt.Errorf("hash: invalid -mode %q (want auto|file|tree)", *mode)
	}

	var id string
	if useTree {
		if id, err = blake3store.HashTree(path); err != nil {
			return fmt.Errorf("hash: tree %s: %w", path, err)
		}
	} else if id, err = blake3store.HashFile(path); err != nil {
		return fmt.Errorf("hash: file %s: %w", path, err)
	}

	_, err = fmt.Fprintln(stdout, id)
	return err
}
