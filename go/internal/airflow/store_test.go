package airflow

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/glusman-team/dakp/go/internal/blake3store"
)

func TestRegisterRecordsSinglePassHashes(t *testing.T) {
	workdir := t.TempDir()
	p := filepath.Join(workdir, "out.tsv")
	if err := os.WriteFile(p, []byte("a\tb\n1\t2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	store := Store{Workdir: workdir}
	ref, err := store.Register(RegisterInput{Path: p, MediaType: TSVMediaType, Rows: 1})
	if err != nil {
		t.Fatalf("Register: %v", err)
	}

	wantID, err := blake3store.HashFile(p)
	if err != nil {
		t.Fatal(err)
	}
	wantSRI, err := blake3store.SHA256SRI(p)
	if err != nil {
		t.Fatal(err)
	}
	if ref.Blake3 != wantID {
		t.Errorf("ref.Blake3 = %q, want %q", ref.Blake3, wantID)
	}
	if ref.Manifest == nil {
		t.Fatal("ref.Manifest is nil")
	}
	m, err := blake3store.ReadManifest(*ref.Manifest)
	if err != nil {
		t.Fatalf("ReadManifest: %v", err)
	}
	if m.Hash.File == nil || *m.Hash.File != wantID {
		t.Errorf("manifest hash.file = %v, want %q", m.Hash.File, wantID)
	}
	if m.Hash.SHA256SRI == nil || *m.Hash.SHA256SRI != wantSRI {
		t.Errorf("manifest hash.sha256_sri = %v, want %q", m.Hash.SHA256SRI, wantSRI)
	}
}
