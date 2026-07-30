package registry

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
)

// resetCommands clears the package-global registry so tests do not leak registrations
// into one another. Test-only; lives in the internal test package so production code
// stays clean.
func resetCommands() {
	mu.Lock()
	defer mu.Unlock()
	cmds = map[string]Command{}
}

func TestRegisterAndDispatch(t *testing.T) {
	resetCommands()
	var gotArgs []string
	Register("greet", func(_ context.Context, args []string) error {
		gotArgs = args
		return nil
	})

	if err := Dispatch(context.Background(), []string{"prog", "greet", "x", "y"}); err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	if strings.Join(gotArgs, ",") != "x,y" {
		t.Errorf("command received args %v, want [x y]", gotArgs)
	}
}

func TestDispatchPropagatesCommandError(t *testing.T) {
	resetCommands()
	sentinel := errors.New("boom")
	Register("fail", func(_ context.Context, _ []string) error { return sentinel })
	if err := Dispatch(context.Background(), []string{"prog", "fail"}); !errors.Is(err, sentinel) {
		t.Errorf("Dispatch error = %v, want %v", err, sentinel)
	}
}

func TestDispatchUnknownCommand(t *testing.T) {
	resetCommands()
	err := Dispatch(context.Background(), []string{"prog", "nope"})
	if !errors.Is(err, ErrUnknownCommand) {
		t.Errorf("error = %v, want ErrUnknownCommand", err)
	}
	if !strings.Contains(err.Error(), "nope") {
		t.Errorf("error should name the bad command: %v", err)
	}
}

func TestDispatchNoCommand(t *testing.T) {
	resetCommands()
	for _, args := range [][]string{{"prog"}, {}, {"prog", ""}} {
		if err := Dispatch(context.Background(), args); !errors.Is(err, ErrNoCommand) {
			t.Errorf("Dispatch(%v) error = %v, want ErrNoCommand", args, err)
		}
	}
}

func TestDispatchHelp(t *testing.T) {
	resetCommands()
	for _, name := range []string{"help", "-h", "--help"} {
		if err := Dispatch(context.Background(), []string{"prog", name}); !errors.Is(err, errHelpRequested) {
			t.Errorf("Dispatch(%s) error = %v, want errHelpRequested", name, err)
		}
	}
}

func TestDuplicateRegisterPanics(t *testing.T) {
	resetCommands()
	Register("dup", func(_ context.Context, _ []string) error { return nil })
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on duplicate registration")
		}
		resetCommands()
	}()
	Register("dup", func(_ context.Context, _ []string) error { return nil })
}

func TestNamesSorted(t *testing.T) {
	resetCommands()
	noop := func(_ context.Context, _ []string) error { return nil }
	Register("charlie", noop)
	Register("alpha", noop)
	Register("bravo", noop)
	got := Names()
	if strings.Join(got, ",") != "alpha,bravo,charlie" {
		t.Errorf("Names = %v, want sorted", got)
	}
}

func TestRunExitCodes(t *testing.T) {
	resetCommands()
	Register("ok", func(_ context.Context, _ []string) error { return nil })
	Register("bad", func(_ context.Context, _ []string) error { return errors.New("kaboom") })

	cases := []struct {
		name     string
		args     []string
		wantCode int
		wantOut  string // substring required on stdout
		wantErr  string // substring required on stderr
	}{
		{"success", []string{"prog", "ok"}, 0, "", ""},
		{"command_error", []string{"prog", "bad"}, 1, "", "kaboom"},
		{"unknown", []string{"prog", "missing"}, 2, "", "unknown command"},
		{"no_command", []string{"prog"}, 2, "", "no command given"},
		{"help", []string{"prog", "help"}, 0, "Commands:", ""},
	}
	for _, c := range cases {
		var stdout, stderr bytes.Buffer
		if code := Run(context.Background(), c.args, &stdout, &stderr); code != c.wantCode {
			t.Errorf("%s: exit code = %d, want %d (stderr=%q)", c.name, code, c.wantCode, stderr.String())
		}
		if c.wantOut != "" && !strings.Contains(stdout.String(), c.wantOut) {
			t.Errorf("%s: stdout %q missing %q", c.name, stdout.String(), c.wantOut)
		}
		if c.wantErr != "" && !strings.Contains(stderr.String(), c.wantErr) {
			t.Errorf("%s: stderr %q missing %q", c.name, stderr.String(), c.wantErr)
		}
	}
}

func TestUsageListsCommands(t *testing.T) {
	resetCommands()
	Register("hash", func(_ context.Context, _ []string) error { return nil })
	var buf bytes.Buffer
	Usage(&buf, "dakp-worker")
	out := buf.String()
	if !strings.Contains(out, "Usage: dakp-worker <command>") {
		t.Errorf("usage header missing: %q", out)
	}
	if !strings.Contains(out, "hash") {
		t.Errorf("usage should list registered command: %q", out)
	}
}
