package blake3store

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// BenchmarkHashTree hashes a synthetic tree of 200 files x 5 MiB, the shape of a
// mid-size interim/ directory, to measure the pipelined read-ahead in hashTreeChunked.
func BenchmarkHashTree(b *testing.B) {
	root := b.TempDir()
	payload := make([]byte, 5<<20)
	for i := range payload {
		payload[i] = byte(i * 31)
	}
	for i := 0; i < 200; i++ {
		p := filepath.Join(root, fmt.Sprintf("f%04d.bin", i))
		if err := os.WriteFile(p, payload, 0o644); err != nil {
			b.Fatal(err)
		}
	}
	b.SetBytes(200 * int64(len(payload)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := HashTree(root); err != nil {
			b.Fatal(err)
		}
	}
}
