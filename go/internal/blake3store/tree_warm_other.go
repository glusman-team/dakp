//go:build !linux

package blake3store

// warmTreeFile is a no-op off Linux: the readahead hint is a best-effort optimization
// only, and golang.org/x/sys/unix exposes Fadvise only where the syscall exists.
func warmTreeFile(string) {}
