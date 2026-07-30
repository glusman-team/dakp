package airflow

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// StageInputs materializes the input artifacts' files into dir (hardlink, copy on failure),
// preserving basenames — the Go parsers classify inputs by filename (DailyMed .xml/.xml.gz, FAERS
// family stems, Drugs@FDA table stems). Mirrors Python go_runner.stage_inputs. On a basename
// collision the file is prefixed with its zero-padded index so distinct inputs never clobber.
func StageInputs(refs []ArtifactRef, dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	for i, ref := range refs {
		base := filepath.Base(ref.URI)
		dest := filepath.Join(dir, base)
		if _, err := os.Lstat(dest); err == nil {
			dest = filepath.Join(dir, fmt.Sprintf("%04d_%s", i, base))
		}
		if err := os.Link(ref.URI, dest); err != nil {
			if cerr := copyFile(ref.URI, dest); cerr != nil {
				return cerr
			}
		}
	}
	return nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}
