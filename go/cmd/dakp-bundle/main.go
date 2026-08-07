// Command dakp-bundle is DAKP's native Airflow Go SDK bundle.
//
// It replaces the old subprocess model (Python shelling out to the dakp-worker CLI via
// workers/go_runner.py) with native Airflow Go Task SDK workers: the heavy parsing/extraction for
// DailyMed / FAERS / Drugs@FDA runs *inside* this bundle, which the Airflow worker's
// ExecutableCoordinator forks once per task instance (coordinator deployment mode). The bundle
// speaks the supervisor wire protocol directly (msgpack-over-IPC) and has native access to the
// Airflow model (XCom / Variables) through sdk.Client.
//
// The DAG structure and task dependencies still live in Python (dags/dakp_build.py) as
// @task.stub(queue="golang") declarations — a documented Go SDK limitation (the Execution API does
// not yet carry DAG structure for non-Python languages). This bundle registers the Go
// implementations of those stub tasks; the dag_id ("dakp_build") and each task_id MUST match the
// Python stub DAG. Task ids are set explicitly via AddTaskWithName so the Go functions keep
// idiomatic camelCase names.
//
// Each extract task: reads the run Config from the `dakp_config` Variable, reads its upstream
// acquisition task's list[ArtifactRef] from XCom, runs the parity-verified extractor in
// internal/airflow (parse -> interim parquet + TSV handoff -> BLAKE3 store + manifests), and
// returns the produced list[ArtifactRef] as its return_value XCom for the shaping stage.
//
// Build + pack with the Go SDK packer (see go/README.md and the Makefile):
//
//	go tool airflow-go-pack --output <executables_root>/dakp-bundle ./go/cmd/dakp-bundle
package main

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"time"

	v1 "github.com/apache/airflow/go-sdk/bundle/bundlev1"
	"github.com/apache/airflow/go-sdk/bundle/bundlev1/bundlev1server"
	"github.com/apache/airflow/go-sdk/sdk"

	"github.com/glusman-team/dakp/go/internal/airflow"
)

// dagID must match the Python stub DAG's dag_id (dags/dakp_build.py: DAG_ID).
const dagID = "dakp_build"

// Task ids — must match the @task.stub function names in the Python DAG.
const (
	taskExtractDailyMed = "extract_dailymed"
	taskExtractFAERS    = "extract_faers"
	taskExtractDrugsFDA = "extract_drugsfda"
)

// configVariable is the Airflow Variable (JSON) holding the per-run Config (workdir/profile/...).
const configVariable = "dakp_config"

// returnKey is the XCom key a task's return value is stored under.
const returnKey = "return_value"

// Set by `-ldflags` at build time (the packer forwards go build flags after `--`).
var (
	bundleName    = "dakp_build"
	bundleVersion = "0.0.0"
)

// dakpBundle implements v1.BundleProvider: it declares the bundle version and registers the
// dags/tasks this bundle can run. RegisterDags is the single source of truth for the bundle's
// dag_id/task_id manifest, so the packed metadata can never drift from what the binary executes.
type dakpBundle struct{}

var _ v1.BundleProvider = (*dakpBundle)(nil)

func (m *dakpBundle) GetBundleVersion() v1.BundleInfo {
	return v1.BundleInfo{Name: bundleName, Version: &bundleVersion}
}

func (m *dakpBundle) RegisterDags(dagbag v1.Registry) error {
	dag := dagbag.AddDag(dagID)
	dag.AddTaskWithName(taskExtractDailyMed, extractDailymed)
	dag.AddTaskWithName(taskExtractFAERS, extractFAERS)
	dag.AddTaskWithName(taskExtractDrugsFDA, extractDrugsFDA)
	return nil
}

func main() {
	if err := bundlev1server.Serve(&dakpBundle{}); err != nil {
		log.Fatal(err)
	}
}

// extractFn is the pure, unit-tested extractor signature (internal/airflow).
type extractFn func(context.Context, airflow.Config, []airflow.ArtifactRef) ([]airflow.ArtifactRef, error)

// extractDailymed is the native Go implementation of the DAG's extract_dailymed stub task.
func extractDailymed(ctx sdk.TIRunContext, client sdk.Client, log *slog.Logger) (any, error) {
	return runExtract(ctx, client, log, taskExtractDailyMed, "acquire_dailymed", airflow.ExtractDailyMed)
}

// extractFAERS is the native Go implementation of extract_faers.
func extractFAERS(ctx sdk.TIRunContext, client sdk.Client, log *slog.Logger) (any, error) {
	return runExtract(ctx, client, log, taskExtractFAERS, "acquire_faers", airflow.ExtractFAERS)
}

// extractDrugsFDA is the native Go implementation of extract_drugsfda.
func extractDrugsFDA(ctx sdk.TIRunContext, client sdk.Client, log *slog.Logger) (any, error) {
	return runExtract(ctx, client, log, taskExtractDrugsFDA, "acquire_drugsfda", airflow.ExtractDrugsFDA)
}

// runExtract is the shared SDK adapter: read the run Config (Variable) + the upstream acquisition
// task's ArtifactRefs (XCom), run the pure extractor, and return the produced ArtifactRefs (pushed
// as this task's return_value XCom). Honors ctx cancellation via the extractor's context.
func runExtract(ctx sdk.TIRunContext, client sdk.Client, log *slog.Logger, taskID, upstream string, fn extractFn) (any, error) {
	start := time.Now()
	cfg, err := readConfig(ctx, client)
	if err != nil {
		return nil, err
	}
	airflow.Started(ctx, taskID)
	airflow.Stat(ctx, taskID, "workdir", cfg.Workdir)
	airflow.Stat(ctx, taskID, "profile", cfg.Profile)
	airflow.Stat(ctx, taskID, "threads", cfg.Threads)
	airflow.Stat(ctx, taskID, "quarter_limit", cfg.QuarterLimit)
	airflow.Stat(ctx, taskID, "release_limit", cfg.ReleaseLimit)
	airflow.Stat(ctx, taskID, "force", cfg.Force)
	airflow.Stat(ctx, taskID, "upstream_task", upstream)
	inputs, err := readUpstreamRefs(ctx, client, upstream)
	if err != nil {
		return nil, err
	}
	airflow.Stat(ctx, taskID, "input_refs", len(inputs))
	for i, ref := range inputs {
		airflow.Stat(ctx, taskID, fmt.Sprintf("input[%d].uri", i), ref.URI)
		airflow.Stat(ctx, taskID, fmt.Sprintf("input[%d].blake3", i), ref.Blake3)
		if ref.Rows != nil {
			airflow.Stat(ctx, taskID, fmt.Sprintf("input[%d].rows", i), *ref.Rows)
		}
	}
	refs, err := fn(ctx, cfg, inputs)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", taskID, err)
	}
	airflow.Stat(ctx, taskID, "output_refs", len(refs))
	totalRows := int64(0)
	for _, ref := range refs {
		if ref.Rows != nil {
			totalRows += *ref.Rows
		}
	}
	airflow.Stat(ctx, taskID, "output_rows_total", totalRows)
	airflow.Finished(ctx, taskID, start)
	return airflow.EncodeArtifactRefs(refs), nil
}

// readConfig reads the per-run Config from the `dakp_config` Variable (JSON). GetVariable falls
// back to AIRFLOW_VAR_DAKP_CONFIG, which makes local dev/tests easy.
func readConfig(ctx sdk.TIRunContext, client sdk.Client) (airflow.Config, error) {
	var cfg airflow.Config
	if err := client.UnmarshalJSONVariable(ctx, configVariable, &cfg); err != nil {
		return cfg, fmt.Errorf("read %q variable: %w", configVariable, err)
	}
	if cfg.Workdir == "" {
		return cfg, fmt.Errorf("%q variable missing workdir", configVariable)
	}
	return cfg, nil
}

// readUpstreamRefs reads an upstream task's return_value XCom (a list of ArtifactRef manifests) and
// decodes it. The dag/run ids come from the executing task instance; the upstream task_id is static.
func readUpstreamRefs(ctx sdk.TIRunContext, client sdk.Client, upstream string) ([]airflow.ArtifactRef, error) {
	ti := ctx.TaskInstance()
	v, err := client.GetXCom(ctx, ti.DagID, ti.RunID, upstream, ti.MapIndex, returnKey, nil)
	if err != nil {
		return nil, fmt.Errorf("read upstream %q xcom: %w", upstream, err)
	}
	return airflow.DecodeArtifactRefs(v)
}
