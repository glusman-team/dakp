// Command dakp-bundle is DAKP's native Airflow Go SDK bundle.
//
// It replaces the old subprocess model (Python shelling out to the dakp-worker CLI via
// workers/go_runner.py) with native Airflow Go Task SDK workers: the heavy parsing/extraction
// for DailyMed / FAERS / Drugs@FDA runs *inside* this bundle, which the Airflow worker's
// ExecutableCoordinator forks once per task instance (coordinator deployment mode). The bundle
// speaks the supervisor wire protocol directly (msgpack-over-IPC) and has native access to the
// Airflow model (XCom / Variables / Connections) through sdk.Client.
//
// The DAG structure and task dependencies still live in Python (dags/dakp_build.py) as
// @task.stub(queue="golang") declarations — a documented Go SDK limitation (the Execution API
// does not yet carry DAG structure for non-Python languages). This bundle registers the Go
// implementations of those stub tasks; the dag_id ("dakp_build") and each task_id MUST match
// the Python stub DAG. Task ids are set explicitly via AddTaskWithName so the Go functions can
// keep idiomatic camelCase names.
//
// Build + pack with the Go SDK packer (see go/README.md and the Makefile):
//
//	go tool airflow-go-pack --output <executables_root>/dakp-bundle ./go/cmd/dakp-bundle
//
// The packed bundle is a single self-contained executable: binary + embedded DAG source + a
// metadata footer (the dag_id/task_id manifest) that the coordinator reads without executing it.
// Run with --airflow-metadata to print that manifest (the packer does this internally).
package main

import (
	"log"
	"log/slog"

	v1 "github.com/apache/airflow/go-sdk/bundle/bundlev1"
	"github.com/apache/airflow/go-sdk/bundle/bundlev1/bundlev1server"
	"github.com/apache/airflow/go-sdk/sdk"
)

// dagID must match the Python stub DAG's dag_id (dags/dakp_build.py: DAG_ID).
const dagID = "dakp_build"

// Task ids — must match the @task.stub function names in the Python DAG.
const (
	taskExtractDailyMed = "extract_dailymed"
	taskExtractFAERS    = "extract_faers"
	taskExtractDrugsFDA = "extract_drugsfda"
)

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

// extractDailymed is the native Go implementation of the DAG's extract_dailymed stub task.
//
// Contract (mirrors the Python TaskFlow stage it replaces): consume the upstream acquisition
// task's list[ArtifactRef] (raw SPL .xml/.xml.gz shards) from XCom, parse them with
// internal/dailymed, finalize the five normalized interim parquet tables into the BLAKE3
// content-addressed store (with manifests), and push the resulting list[ArtifactRef] as this
// task's return_value XCom for the shaping stage.
//
// NOTE(skeleton): this is the SDK-wiring stub. The real D1 body (XCom in -> stage -> parse ->
// parquet + store + manifest -> XCom out) lands in the next step; for now it logs the task
// runtime context and returns a placeholder so the bundle builds, packs, and registers.
func extractDailymed(ctx sdk.TIRunContext, client sdk.Client, log *slog.Logger) (any, error) {
	return runExtractStub(ctx, log, taskExtractDailyMed)
}

// extractFAERS is the native Go implementation of extract_faers (FAERS ASCII quarters ->
// faers_cases interim table). See extractDailymed for the shared contract.
func extractFAERS(ctx sdk.TIRunContext, client sdk.Client, log *slog.Logger) (any, error) {
	return runExtractStub(ctx, log, taskExtractFAERS)
}

// extractDrugsFDA is the native Go implementation of extract_drugsfda (Drugs@FDA tab-delimited
// tables -> products/applications/submissions/lookups interim tables). See extractDailymed.
func extractDrugsFDA(ctx sdk.TIRunContext, client sdk.Client, log *slog.Logger) (any, error) {
	return runExtractStub(ctx, log, taskExtractDrugsFDA)
}

// runExtractStub logs the task runtime context (proving the sdk.TIRunContext injection works)
// and returns a placeholder result. Replaced by the real D1 extraction body per task.
func runExtractStub(ctx sdk.TIRunContext, log *slog.Logger, taskID string) (any, error) {
	ti, dagRun := ctx.TaskInstance(), ctx.DagRun()
	log.InfoContext(ctx, "dakp extract task invoked",
		"task_id", taskID,
		"dag_id", ti.DagID,
		"run_id", ti.RunID,
		"try_number", ti.TryNumber,
		"logical_date", dagRun.LogicalDate,
	)
	return map[string]any{"task_id": taskID, "status": "skeleton"}, nil
}
