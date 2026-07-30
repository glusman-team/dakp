// Command dakp-worker is DAKP's native Go worker CLI. Heavy parsing/extraction workers
// (FAERS ASCII parsing, DailyMed XML shard extraction, BLAKE3 tree hashing, high-volume
// text normalization) run here as subcommands; Airflow tasks call this binary (or
// `go run ./cmd/dakp-worker` in development) and stream its stdout (artifact ids /
// manifests) and stderr (structured JSON logs) into the task logger.
//
// Subcommands self-register via init() in their own files in this package (package main),
// so adding a new extractor is a NEW file — main.go never changes and independent
// extractor workers merge cleanly in parallel. See internal/registry and go/README.md.
package main

import (
	"os"

	"github.com/glusman-team/dakp/go/internal/registry"
)

func main() {
	registry.Main(os.Args)
}
