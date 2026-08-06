package airflow

import (
	"container/heap"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/glusman-team/dakp/go/internal/faers"
	"github.com/parquet-go/parquet-go"
)

// This file implements the bounded-memory FAERS case pipeline pieces the streaming
// ExtractFAERS orchestration uses: per-quarter sorted "kept-case run" files plus a k-way
// merge that reproduces the batch path's global ordering without ever holding all cases in
// memory at once (see plans/fix-faers-memory.md).

// faersRunColumns is the on-disk schema of per-quarter kept-case run files: the public
// faers_cases.tsv columns in contract order plus drug_seq — an ordering-only column the
// global merge needs to reproduce the batch path's sort (cases.parquet emits it empty; the
// public TSV excludes it).
var faersRunColumns = append(append([]string{}, faers.CasesTSVColumns...), "drug_seq")

// Sort-key positions inside faersRunColumns (mirrors faers caseSortKey).
const (
	faersRunPrimaryID  = 1
	faersRunIndication = 9
	faersRunDrugSeq    = 11
)

// writeFAERSRun writes one quarter's kept cases as a sorted run file. The input must already
// be sorted by (primaryid, drug_seq, indication) — BuildQuarterCases guarantees that and the
// streaming dedup filter preserves order.
func writeFAERSRun(path string, cases []faers.Case) error {
	w, err := NewStringParquetWriter(path, faersRunColumns)
	if err != nil {
		return err
	}
	row := make([]string, len(faersRunColumns))
	for _, c := range cases {
		row[0], row[1], row[2], row[3] = c.Quarter, c.PrimaryID, c.CaseID, c.Source
		row[4], row[5], row[6], row[7] = c.OccpCod, c.ReporterCountry, c.Drugname, c.Ingredient
		row[8], row[9], row[10], row[11] = c.Nda, c.Indication, c.Effects, c.DrugSeq
		if err := w.Append(row); err != nil {
			w.Close()
			return err
		}
	}
	_, err = w.Close()
	return err
}

// faersRunReader streams the rows of one kept-case run file back in faersRunColumns order.
type faersRunReader struct {
	f         *os.File
	groups    []parquet.RowGroup
	groupIdx  int
	rows      parquet.Rows
	leafToCol []int // parquet leaf index -> faersRunColumns index
	cur       []string
	done      bool
	closed    bool
}

// openFAERSRun opens a run file written by writeFAERSRun for streaming reads.
func openFAERSRun(path string) (*faersRunReader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	pf, err := parquet.OpenFile(f, st.Size())
	if err != nil {
		f.Close()
		return nil, err
	}
	colIdx := make(map[string]int, len(faersRunColumns))
	for i, name := range faersRunColumns {
		colIdx[name] = i
	}
	leafToCol := make([]int, 0, len(faersRunColumns))
	for _, p := range pf.Schema().Columns() {
		ci, ok := colIdx[p[len(p)-1]]
		if !ok {
			f.Close()
			return nil, fmt.Errorf("faers run %s: unexpected column %q", path, p[len(p)-1])
		}
		leafToCol = append(leafToCol, ci)
	}
	return &faersRunReader{f: f, groups: pf.RowGroups(), leafToCol: leafToCol}, nil
}

// next advances to the next row (available via r.cur), returning false at EOF. The cur slice
// is reused across calls — consume it before the next call.
func (r *faersRunReader) next() (bool, error) {
	if r.done {
		return false, nil
	}
	buf := make([]parquet.Row, 1)
	for {
		if r.rows == nil {
			if r.groupIdx >= len(r.groups) {
				r.done = true
				return false, nil
			}
			r.rows = r.groups[r.groupIdx].Rows()
			r.groupIdx++
		}
		n, err := r.rows.ReadRows(buf)
		if n > 0 {
			if r.cur == nil {
				r.cur = make([]string, len(faersRunColumns))
			}
			for i, v := range buf[0] {
				r.cur[r.leafToCol[i]] = v.String()
			}
			return true, nil
		}
		r.rows.Close()
		r.rows = nil
		if err != nil && err != io.EOF {
			return false, err
		}
		// io.EOF (or a zero-row read): move on to the next row group.
	}
}

func (r *faersRunReader) Close() error {
	if r.closed {
		return nil
	}
	r.closed = true
	if r.rows != nil {
		r.rows.Close()
		r.rows = nil
	}
	return r.f.Close()
}

// faersMergeCursor is one run in the k-way merge.
type faersMergeCursor struct {
	r   *faersRunReader
	idx int // run index; lower = NEWER quarter (runs are passed most-recent-first)
}

type faersMergeHeap []*faersMergeCursor

func (h faersMergeHeap) Len() int { return len(h) }

// Less orders merged rows by the case sort key (primaryid, drug_seq, indication), breaking
// ties toward the NEWER run (lower index) — exactly reproducing sort.SliceStable over the
// batch path's most-recent-first concat, so merged output is byte-identical to the batch path.
func (h faersMergeHeap) Less(i, j int) bool {
	ai, bi := h[i].r.cur, h[j].r.cur
	if ai[faersRunPrimaryID] != bi[faersRunPrimaryID] {
		return ai[faersRunPrimaryID] < bi[faersRunPrimaryID]
	}
	if ai[faersRunDrugSeq] != bi[faersRunDrugSeq] {
		return ai[faersRunDrugSeq] < bi[faersRunDrugSeq]
	}
	if ai[faersRunIndication] != bi[faersRunIndication] {
		return ai[faersRunIndication] < bi[faersRunIndication]
	}
	return h[i].idx < h[j].idx
}

func (h faersMergeHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *faersMergeHeap) Push(x any)   { *h = append(*h, x.(*faersMergeCursor)) }
func (h *faersMergeHeap) Pop() any {
	old := *h
	n := len(old)
	c := old[n-1]
	old[n-1] = nil
	*h = old[:n-1]
	return c
}

// mergeFAERSRuns k-way-merges sorted run files — passed MOST-RECENT-FIRST — and hands each
// merged row (faersRunColumns order) to emit in global (primaryid, drug_seq, indication)
// order with newer-quarter tiebreak. The emitted slice is reused between calls; emit must
// consume (or copy) it before returning. Empty runs are skipped.
func mergeFAERSRuns(paths []string, emit func(row []string) error) error {
	var readers []*faersRunReader
	defer func() {
		for _, r := range readers {
			r.Close()
		}
	}()
	h := &faersMergeHeap{}
	for i, p := range paths {
		rd, err := openFAERSRun(p)
		if err != nil {
			return err
		}
		readers = append(readers, rd)
		ok, err := rd.next()
		if err != nil {
			return err
		}
		if ok {
			heap.Push(h, &faersMergeCursor{r: rd, idx: i})
		}
	}
	for h.Len() > 0 {
		c := (*h)[0]
		if err := emit(c.r.cur); err != nil {
			return err
		}
		ok, err := c.r.next()
		if err != nil {
			return err
		}
		if ok {
			heap.Fix(h, 0)
		} else {
			heap.Pop(h)
		}
	}
	return nil
}

// writeFAERSRunTSVRow writes one merged run row as a public faers_cases.tsv data row: the
// contract projection (CasesTSVColumns — the run's trailing drug_seq dropped), tab-joined.
func writeFAERSRunTSVRow(w io.Writer, row []string) error {
	if _, err := io.WriteString(w, strings.Join(row[:len(faers.CasesTSVColumns)], "\t")); err != nil {
		return err
	}
	_, err := io.WriteString(w, "\n")
	return err
}

// faersCaseRow17 projects one merged run row (faersRunColumns order) to the 17-column
// cases.parquet layout (faersCaseColumns): the public fields carry the data, the provenance
// columns (nda_raw/role_cod/drug_seq/indi_drug_seq/source_file/source_record_id) are empty —
// exactly the reconstruction the batch path emitted.
func faersCaseRow17(row []string) []string {
	return []string{
		row[0], row[1], row[2], row[3], row[4], row[5], // quarter..reporter_country
		row[6], row[7], row[8], // drugname, ingredient, nda
		"", "", "", "", // nda_raw, role_cod, drug_seq, indi_drug_seq
		row[9], row[10], // indication, effects
		"", "", // source_file, source_record_id
	}
}
