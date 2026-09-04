//go:build linux

package blake3store

import (
	"os"

	"golang.org/x/sys/unix"
)

// warmTreeFile issues a best-effort async kernel readahead hint for path so the tree
// hasher's upcoming sequential reads hit the page cache. All errors are ignored: the
// hashing reads are authoritative, this only affects WHEN pages are fetched.
func warmTreeFile(path string) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	_ = unix.Fadvise(int(f.Fd()), 0, 0, unix.FADV_WILLNEED) // 0,0 = whole file
}
