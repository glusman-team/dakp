package airflow

import (
	"context"
	"fmt"
	"log/slog"
	"time"
)

// One stat per log line.
//
// Airflow renders each Go SDK slog record as a single JSON line in the task log, so DAKP never
// packs several stats into one record: every stat gets its own record whose EVENT TEXT is the
// self-contained, greppable line "<event>: <key> = <value>" — mirroring the Python
// logging_setup.stats() convention. In coordinator mode the Go SDK installs its socket log
// handler as slog's default logger before task code runs, so package-level slog calls from the
// extractors land in the Airflow task log with no plumbing.

// Stat emits one INFO log line: "<event>: <key> = <value>".
func Stat(ctx context.Context, event, key string, value any) {
	slog.InfoContext(ctx, fmt.Sprintf("%s: %s = %v", event, key, value))
}

// StatDebug emits one DEBUG log line: "<event>: <key> = <value>" (per-artifact detail).
func StatDebug(ctx context.Context, event, key string, value any) {
	slog.DebugContext(ctx, fmt.Sprintf("%s: %s = %v", event, key, value))
}

// Started marks a phase beginning: "<event>: started".
func Started(ctx context.Context, event string) {
	slog.InfoContext(ctx, event+": started")
}

// Finished marks a phase end with its elapsed time, one line each:
// "<event>: finished = true" and "<event>: elapsed_s = <N>".
func Finished(ctx context.Context, event string, start time.Time) {
	slog.InfoContext(ctx, event+": finished = true")
	Stat(ctx, event, "elapsed_s", fmt.Sprintf("%.3f", time.Since(start).Seconds()))
}

// StatOutput narrates one registered output artifact — output name, then path / blake3 / rows /
// schema_fingerprint, each its own line ("<event>: <table>.<key> = <value>").
func StatOutput(ctx context.Context, event, table string, ref ArtifactRef) {
	Stat(ctx, event, "output", table)
	Stat(ctx, event, table+".path", ref.URI)
	Stat(ctx, event, table+".blake3", ref.Blake3)
	if ref.Rows != nil {
		Stat(ctx, event, table+".rows", *ref.Rows)
	}
	if ref.SchemaFingerprint != nil {
		Stat(ctx, event, table+".schema_fingerprint", *ref.SchemaFingerprint)
	}
}
