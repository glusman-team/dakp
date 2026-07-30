// Package registry is a tiny self-registration command dispatcher for the dakp-worker
// CLI. Subcommands register themselves from init() in their own files (all in package
// main under go/cmd/dakp-worker), so adding a new extractor subcommand is a NEW file that
// never requires editing main.go or this package — independent extractor workers merge
// cleanly in parallel. See go/README.md for the step-by-step pattern.
package registry

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"sync"
)

// Command is a subcommand implementation. args excludes the program name and the
// subcommand name (it receives only the flags/positionals after the subcommand).
type Command func(ctx context.Context, args []string) error

var (
	mu   sync.RWMutex
	cmds = map[string]Command{}
)

// Register adds a named subcommand. It panics on an empty name, a nil func, or a duplicate
// name — all programming errors that should fail loudly at startup.
func Register(name string, fn Command) {
	if name == "" {
		panic("registry: command name must be non-empty")
	}
	if fn == nil {
		panic("registry: command func must be non-nil: " + name)
	}
	mu.Lock()
	defer mu.Unlock()
	if _, dup := cmds[name]; dup {
		panic("registry: duplicate command registration: " + name)
	}
	cmds[name] = fn
}

// Lookup returns the command registered under name.
func Lookup(name string) (Command, bool) {
	mu.RLock()
	defer mu.RUnlock()
	fn, ok := cmds[name]
	return fn, ok
}

// Names returns the registered command names, sorted.
func Names() []string {
	mu.RLock()
	defer mu.RUnlock()
	names := make([]string, 0, len(cmds))
	for n := range cmds {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// Sentinel dispatch errors.
var (
	// ErrNoCommand indicates no subcommand was supplied.
	ErrNoCommand = errors.New("no command given")
	// ErrUnknownCommand wraps the name of an unregistered subcommand.
	ErrUnknownCommand = errors.New("unknown command")
	// errHelpRequested signals that usage should be printed (exit 0).
	errHelpRequested = errors.New("help requested")
)

func isHelp(name string) bool {
	switch name {
	case "help", "-h", "--help":
		return true
	}
	return false
}

// Dispatch routes args to the registered command. args[0] is the program name and args[1]
// is the subcommand (mirroring os.Args). It returns ErrNoCommand if no subcommand is
// present, an ErrUnknownCommand-wrapped error for an unregistered name, errHelpRequested
// for help flags, or the command's own error. It writes nothing and never calls os.Exit —
// Run/Main own output and exit codes, which keeps Dispatch unit-testable.
func Dispatch(ctx context.Context, args []string) error {
	if len(args) < 2 || args[1] == "" {
		return ErrNoCommand
	}
	name := args[1]
	if isHelp(name) {
		return errHelpRequested
	}
	fn, ok := Lookup(name)
	if !ok {
		return fmt.Errorf("%w: %s", ErrUnknownCommand, name)
	}
	return fn(ctx, args[2:])
}

// Usage writes the command list to w.
func Usage(w io.Writer, prog string) {
	if prog == "" {
		prog = "dakp-worker"
	}
	fmt.Fprintf(w, "Usage: %s <command> [args]\n\nCommands:\n", prog)
	for _, n := range Names() {
		fmt.Fprintf(w, "  %s\n", n)
	}
}

// Run dispatches args (typically os.Args) and returns a process exit code:
//
//	0 on success or help
//	1 on a command execution error
//	2 on usage errors (no command / unknown command)
//
// It writes usage/errors to the given writers and never calls os.Exit, so it is testable.
func Run(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	prog := "dakp-worker"
	if len(args) > 0 && args[0] != "" {
		prog = args[0]
	}
	err := Dispatch(ctx, args)
	switch {
	case err == nil:
		return 0
	case errors.Is(err, errHelpRequested):
		Usage(stdout, prog)
		return 0
	case errors.Is(err, ErrNoCommand), errors.Is(err, ErrUnknownCommand):
		fmt.Fprintln(stderr, "error:", err)
		Usage(stderr, prog)
		return 2
	default:
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
}

// Main dispatches args (typically os.Args) and exits the process with the code from Run.
// cmd/dakp-worker's main calls registry.Main(os.Args) and nothing else.
func Main(args []string) {
	os.Exit(Run(context.Background(), args, os.Stdout, os.Stderr))
}
