// Package dailymed is the native Go DailyMed SPL XML extractor. It is a faithful port of
// the Python reference extractor in src/dakp_pipeline/extract/spl_xml.py: the same
// streaming, gzip-aware, namespace-robust parse of HL7 v3 SPL batches, the same normalized
// interim tables (documents, sets, approvals, ingredients, LOINC-coded sections), and the
// same BLAKE3 source_record_id derivation (via internal/pipeline.SourceRecordID, which is
// parity-locked to spl_xml._source_record_id).
//
// Parsing streams tokens from encoding/xml and builds an in-memory tree for ONE <document>
// at a time (freed after each document), so memory is constant per document rather than per
// file — mirroring the Python iterparse + elem.clear() approach. These SPL batches are
// huge; the whole file is never materialized.
//
// Two document shapes are supported, exactly as in Python:
//
//   - mock — the namespace-free simplified fixture (direct <setId> / <activeIngredient> /
//     <section loinc=...> children);
//   - HL7 v3 — real DailyMed SPL (urn:hl7-org:v3): set id from <setId root=>, approvals
//     from subjectOf/approval (NDA ids under the FDA OID, application-type codes under the
//     HL7 code system), active/inactive ingredients from activeMoiety /
//     inactiveIngredientSubstance subtrees, and LOINC-coded sections.
//
// Output is uncompressed TSV (the Tablassert-facing handoff), with a polars-compatible
// quoting rule so Go and Python write byte-identical tables for the same rows.
package dailymed

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"golang.org/x/sync/errgroup"

	"github.com/glusman-team/dakp/go/internal/blake3store"
	"github.com/glusman-team/dakp/go/internal/pipeline"
)

// Column contracts for the normalized tables. These mirror, in order and name, the Python
// column lists in src/dakp_pipeline/extract/spl_xml.py (SPL_*_COLUMNS) and
// src/dakp_pipeline/io/schemas.py (DAILYMED_SPL_DOCUMENTS_COLUMNS), so the Go TSV headers
// are identical to the Python ones.
var (
	// DocumentsColumns is the wide, denormalized public contract: one row per section with
	// the document's first active ingredient and first approval denormalized onto it.
	DocumentsColumns = []string{
		"spl_document_id",
		"spl_set_id",
		"xml_path",
		"release_file",
		"approval_code",
		"approval_type",
		"loinc_code",
		"section_name",
		"section_text",
		"active_ingredient_name",
		"active_ingredient_unii",
	}
	// SetsColumns is one row per distinct SPL set id.
	SetsColumns = []string{"source_record_id", "spl_set_id", "release_file", "xml_path"}
	// ApprovalsColumns is one row per approval (id/code/type) per set.
	ApprovalsColumns = []string{"source_record_id", "spl_set_id", "approval_id", "approval_code", "approval_type", "release_file", "xml_path"}
	// IngredientsColumns is active + inactive ingredients, each with UNII + role.
	IngredientsColumns = []string{"source_record_id", "spl_set_id", "ingredient_name", "ingredient_unii", "role", "release_file", "xml_path"}
	// SectionsColumns is the proper per-section table: LOINC code, title, raw + clean text.
	SectionsColumns = []string{
		"source_record_id",
		"spl_document_id",
		"spl_set_id",
		"loinc_code",
		"section_name",
		"section_title",
		"raw_text",
		"clean_text",
		"release_file",
		"xml_path",
	}
)

// SectionCodeNames maps the LOINC section codes DAKP consumes to stable output names
// (mirrors spl_xml.SECTION_CODE_NAMES). Codes absent here fall back to the XML name
// attribute or the LOINC code itself.
var SectionCodeNames = map[string]string{
	"34067-9": "indications_and_usage",
	"34070-3": "contraindications",
	"34066-1": "boxed_warning",
	"42229-5": "warnings_and_precautions",
}

// Approval OID roots / code systems ported from the legacy parser (mirrors spl_xml).
const (
	ndaOID         = "2.16.840.1.113883.3.150"    // subjectOf/approval/id[@root] -> extension (NDA id)
	applCodeSystem = "2.16.840.1.113883.3.26.1.1" // subjectOf/approval/code[@codeSystem] -> application type
	hl7v3Namespace = "urn:hl7-org:v3"             // real DailyMed SPL namespace
	headPeekBytes  = 4096                         // bytes peeked to detect the HL7 v3 namespace
)

// Approval is one parsed approval (NDA id + application-type code).
type Approval struct {
	ID   string
	Code string
	Type string
}

// Ingredient is one active or inactive ingredient with its UNII and role.
type Ingredient struct {
	Name string
	UNII string
	Role string // "active" | "inactive"
}

// Section is one LOINC-coded section with its raw and whitespace-collapsed text.
type Section struct {
	LOINC     string
	Name      string
	Title     string
	RawText   string
	CleanText string
}

// DocumentRecord is a fully parsed SPL document, before flattening into normalized tables.
type DocumentRecord struct {
	SetID       string
	Approvals   []Approval
	Ingredients []Ingredient
	Sections    []Section
	Warnings    []string
}

// --- generic element tree (one document at a time) -----------------------------

// node is a lightweight XML element capturing everything the extractor needs: the local
// name, attributes keyed by local name, and an ORDERED mix of text chunks and child
// elements. The ordered kids preserve ElementTree itertext() semantics (text interleaved
// with children in document order), which the section text extraction relies on.
type node struct {
	name  string
	attrs map[string]string
	kids  []kid
}

// kid is one ordered child of a node: either a text chunk (elem == nil) or a child element.
type kid struct {
	text string
	elem *node
}

// attr returns the whitespace-trimmed value of the attribute with the given local name
// ("" if absent), mirroring spl_xml._attr.
func (n *node) attr(key string) string { return strings.TrimSpace(n.attrs[key]) }

// elemText returns the element's leading text (the text before its first child element),
// mirroring ElementTree's Element.text used by findtext() in the mock set-id path.
func (n *node) elemText() string {
	var sb strings.Builder
	for _, k := range n.kids {
		if k.elem != nil {
			break
		}
		sb.WriteString(k.text)
	}
	return sb.String()
}

// writeText appends all descendant text in document order to sb (itertext semantics).
func (n *node) writeText(sb *strings.Builder) {
	for _, k := range n.kids {
		if k.elem != nil {
			k.elem.writeText(sb)
		} else {
			sb.WriteString(k.text)
		}
	}
}

// concatText returns all descendant text in document order (no stripping/collapsing).
func (n *node) concatText() string {
	var sb strings.Builder
	n.writeText(&sb)
	return sb.String()
}

// text returns the whitespace-collapsed descendant text (mirrors spl_xml._text).
func (n *node) text() string { return collapseWS(n.concatText()) }

// allText returns the descendant text with only leading/trailing whitespace stripped,
// preserving internal whitespace (mirrors spl_xml._all_text).
func (n *node) allText() string { return strings.TrimSpace(n.concatText()) }

// iter returns n and all its descendants whose local name matches name, in pre-order
// document order — exactly ElementTree's Element.iter() filtering semantics.
func (n *node) iter(name string) []*node {
	var out []*node
	var walk func(*node)
	walk = func(x *node) {
		if x.name == name {
			out = append(out, x)
		}
		for _, k := range x.kids {
			if k.elem != nil {
				walk(k.elem)
			}
		}
	}
	walk(n)
	return out
}

// directChildren returns the immediate child elements whose local name matches name, in
// order (mirrors spl_xml._direct_children).
func (n *node) directChildren(name string) []*node {
	var out []*node
	for _, k := range n.kids {
		if k.elem != nil && k.elem.name == name {
			out = append(out, k.elem)
		}
	}
	return out
}

// decodeNode reads one element subtree from the streaming decoder into a node tree. It
// consumes tokens up to and including the matching EndElement. Comments, processing
// instructions, and directives are dropped (ElementTree's itertext ignores them too).
func decodeNode(d *xml.Decoder, start xml.StartElement) (*node, error) {
	n := &node{name: start.Name.Local, attrs: make(map[string]string, len(start.Attr))}
	for _, a := range start.Attr {
		n.attrs[a.Name.Local] = a.Value
	}
	for {
		tok, err := d.Token()
		if err != nil {
			return nil, err // io.EOF here means a truncated/malformed document
		}
		switch t := tok.(type) {
		case xml.StartElement:
			child, err := decodeNode(d, t)
			if err != nil {
				return nil, err
			}
			n.kids = append(n.kids, kid{elem: child})
		case xml.EndElement:
			return n, nil
		case xml.CharData:
			// CharData is reused by the decoder; string(t) copies it.
			n.kids = append(n.kids, kid{text: string(t)})
		}
	}
}

// --- streaming parse -----------------------------------------------------------

// parseDocuments streams SPL <document> elements from r, parsing each into a
// DocumentRecord. Only one document tree is held at a time (constant memory per document).
func parseDocuments(r io.Reader, isHL7v3 bool) ([]DocumentRecord, error) {
	d := xml.NewDecoder(r)
	var docs []DocumentRecord
	for {
		tok, err := d.Token()
		if err == io.EOF {
			return docs, nil
		}
		if err != nil {
			return nil, err
		}
		se, ok := tok.(xml.StartElement)
		if !ok || se.Name.Local != "document" {
			continue
		}
		n, err := decodeNode(d, se)
		if err != nil {
			return nil, err
		}
		if isHL7v3 {
			docs = append(docs, parseHL7v3Document(n))
		} else {
			docs = append(docs, parseMockDocument(n))
		}
	}
}

// gzReadCloser closes both the gzip reader and its underlying file.
type gzReadCloser struct {
	gz *gzip.Reader
	f  *os.File
}

func (g *gzReadCloser) Read(p []byte) (int, error) { return g.gz.Read(p) }

func (g *gzReadCloser) Close() error {
	gzErr := g.gz.Close()
	if err := g.f.Close(); err != nil {
		return err
	}
	return gzErr
}

// openSPL opens an SPL file as a byte stream, transparently gunzipping .gz inputs
// (mirrors spl_xml._open_spl).
func openSPL(path string) (io.ReadCloser, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	if strings.HasSuffix(strings.ToLower(path), ".gz") {
		gz, err := gzip.NewReader(f)
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("gunzip %s: %w", path, err)
		}
		return &gzReadCloser{gz: gz, f: f}, nil
	}
	return f, nil
}

// detectHL7v3 peeks the head of the (decompressed) file to detect the HL7 v3 namespace,
// distinguishing real DailyMed SPL from the namespace-free mock fixture (mirrors
// spl_xml._looks_hl7v3).
func detectHL7v3(path string) (bool, error) {
	rc, err := openSPL(path)
	if err != nil {
		return false, err
	}
	defer rc.Close()
	head := make([]byte, headPeekBytes)
	n, _ := io.ReadFull(rc, head)
	return bytes.Contains(head[:n], []byte(hl7v3Namespace)), nil
}

// ParseFile streams one SPL file (gzipped or plain) into its DocumentRecords, auto-
// detecting the HL7 v3 vs mock shape.
func ParseFile(path string) ([]DocumentRecord, error) {
	isHL7v3, err := detectHL7v3(path)
	if err != nil {
		return nil, err
	}
	rc, err := openSPL(path)
	if err != nil {
		return nil, err
	}
	defer rc.Close()
	return parseDocuments(rc, isHL7v3)
}

// --- mock (namespace-free) document parse --------------------------------------

func parseMockDocument(n *node) DocumentRecord {
	rec := DocumentRecord{}
	if ids := n.directChildren("setId"); len(ids) > 0 {
		rec.SetID = strings.TrimSpace(ids[0].elemText())
	}
	if rec.SetID == "" {
		rec.Warnings = append(rec.Warnings, "document missing setId")
	}
	for _, ap := range n.directChildren("approval") {
		code := ap.attr("code")
		rec.Approvals = append(rec.Approvals, Approval{ID: code, Code: code, Type: ap.attr("type")})
	}
	for _, ing := range n.directChildren("activeIngredient") {
		rec.Ingredients = append(rec.Ingredients, Ingredient{Name: ing.attr("name"), UNII: unii(ing.attr("unii")), Role: "active"})
	}
	for _, ing := range n.directChildren("inactiveIngredient") {
		rec.Ingredients = append(rec.Ingredients, Ingredient{Name: ing.attr("name"), UNII: unii(ing.attr("unii")), Role: "inactive"})
	}
	rec.Sections = collectSections(n.iter("section"), &rec.Warnings)
	return rec
}

// --- HL7 v3 (real DailyMed) document parse -------------------------------------

func parseHL7v3Document(n *node) DocumentRecord {
	rec := DocumentRecord{}
	for _, sid := range n.iter("setId") {
		if root := sid.attr("root"); root != "" {
			rec.SetID = strings.ToLower(root)
			break
		}
	}
	if rec.SetID == "" {
		rec.Warnings = append(rec.Warnings, "document missing setId root")
	}
	rec.Approvals = hl7v3Approvals(n)
	rec.Ingredients = hl7v3Ingredients(n)
	rec.Sections = collectSections(n.iter("section"), &rec.Warnings)
	return rec
}

// hl7v3Approvals ports spl_xml._hl7v3_approvals: one row per distinct NDA id (insertion
// order), carrying the application-type code. The returned type is the LAST code seen for
// that NDA id (mirroring the legacy type_by_id overwrite behavior).
func hl7v3Approvals(n *node) []Approval {
	var order []string
	seen := map[string]bool{}
	typeByID := map[string]string{}
	for _, approval := range n.iter("approval") {
		ndaID := ""
		for _, ident := range approval.iter("id") {
			if ident.attr("root") == ndaOID {
				ndaID = ident.attr("extension")
				break
			}
		}
		applType := ""
		for _, code := range approval.iter("code") {
			if code.attr("codeSystem") == applCodeSystem {
				applType = code.attr("code")
				break
			}
		}
		if ndaID == "" {
			continue
		}
		if !seen[ndaID] {
			seen[ndaID] = true
			order = append(order, ndaID)
		}
		typeByID[ndaID] = applType
	}
	out := make([]Approval, 0, len(order))
	for _, id := range order {
		out = append(out, Approval{ID: id, Code: id, Type: typeByID[id]})
	}
	return out
}

// hl7v3Ingredients ports spl_xml._hl7v3_ingredients: active ingredients from activeMoiety
// subtrees and inactive ingredients from inactiveIngredientSubstance / ingredient
// [@classCode=IACT] subtrees, deduplicated by (role, unii, lower(name)) in traversal order.
func hl7v3Ingredients(n *node) []Ingredient {
	var out []Ingredient
	seen := map[[3]string]bool{}
	add := func(substance *node, role string) {
		name, code := "", ""
		for _, k := range substance.kids {
			if k.elem == nil {
				continue
			}
			switch k.elem.name {
			case "name":
				name = k.elem.text()
			case "code":
				code = k.elem.attr("code")
			}
		}
		u := unii(code)
		key := [3]string{role, u, strings.ToLower(name)}
		if seen[key] || name == "" {
			return
		}
		seen[key] = true
		out = append(out, Ingredient{Name: name, UNII: u, Role: role})
	}
	for _, substance := range n.iter("activeMoiety") {
		// activeMoiety wraps an inner activeMoiety / activeIngredientSubstance element.
		for _, inner := range substance.iter("activeMoiety") {
			add(inner, "active")
		}
		for _, inner := range substance.iter("activeIngredientSubstance") {
			add(inner, "active")
		}
	}
	for _, substance := range n.iter("inactiveIngredientSubstance") {
		add(substance, "inactive")
	}
	for _, ing := range n.iter("ingredient") {
		if ing.attr("classCode") == "IACT" {
			for _, inner := range ing.iter("ingredientSubstance") {
				add(inner, "inactive")
			}
		}
	}
	return out
}

// collectSections builds Section records from <section> elements (shared by both shapes),
// mirroring spl_xml._collect_sections.
func collectSections(sectionElems []*node, warnings *[]string) []Section {
	sections := make([]Section, 0, len(sectionElems))
	for _, sec := range sectionElems {
		loinc := ""
		for _, code := range sec.iter("code") {
			candidate := code.attr("code")
			if candidate == "" {
				continue
			}
			// Prefer the first code that looks like a LOINC; the `loinc == ""` fallback
			// means the first non-empty code wins (mirrors the Python branch exactly).
			if (loinc == "" && looksLoinc(candidate)) || loinc == "" {
				loinc = candidate
			}
			break
		}
		if loinc == "" {
			loinc = sec.attr("loinc") // mock stores LOINC directly on <section>
		}
		name := sec.attr("name")
		if name == "" {
			if mapped, ok := SectionCodeNames[loinc]; ok {
				name = mapped
			} else {
				name = loinc
			}
		}
		title := ""
		if titles := sec.iter("title"); len(titles) > 0 {
			title = titles[0].text()
		}
		if title == "" {
			title = name
		}
		rawText := sec.allText()
		cleanText := collapseWS(rawText)
		if loinc == "" {
			*warnings = append(*warnings, "section missing LOINC code")
		}
		sections = append(sections, Section{LOINC: loinc, Name: name, Title: title, RawText: rawText, CleanText: cleanText})
	}
	return sections
}

// --- table builders ------------------------------------------------------------

// documentID derives the per-section document id: "<set>#<loinc>" when a LOINC is present,
// else the bare set id (mirrors spl_xml._document_rows / _section_rows).
func documentID(setID, loinc string) string {
	if loinc != "" {
		return setID + "#" + loinc
	}
	return setID
}

// DocumentRows builds the wide public-contract rows for one document: one row per section
// with the first active ingredient and first approval denormalized (mirrors
// spl_xml._document_rows).
func DocumentRows(rec DocumentRecord, releaseFile string) [][]string {
	ingName, ingUNII := "", ""
	for _, ing := range rec.Ingredients {
		if ing.Role == "active" {
			ingName, ingUNII = ing.Name, ing.UNII
			break
		}
	}
	apCode, apType := "", ""
	if len(rec.Approvals) > 0 {
		apCode, apType = rec.Approvals[0].Code, rec.Approvals[0].Type
	}
	rows := make([][]string, 0, len(rec.Sections))
	for _, sec := range rec.Sections {
		rows = append(rows, []string{
			documentID(rec.SetID, sec.LOINC),
			rec.SetID,
			releaseFile,
			releaseFile,
			apCode,
			apType,
			sec.LOINC,
			sec.Name,
			sec.CleanText,
			ingName,
			ingUNII,
		})
	}
	return rows
}

// SetRows builds the set rows for one document (nil when the document has no set id),
// mirroring spl_xml._set_rows.
func SetRows(rec DocumentRecord, sourceID, releaseFile string) [][]string {
	if rec.SetID == "" {
		return nil
	}
	return [][]string{{
		pipeline.SourceRecordID(sourceID, "set", rec.SetID),
		rec.SetID,
		releaseFile,
		releaseFile,
	}}
}

// ApprovalRows builds the approval rows for one document, mirroring spl_xml._approval_rows.
func ApprovalRows(rec DocumentRecord, sourceID, releaseFile string) [][]string {
	rows := make([][]string, 0, len(rec.Approvals))
	for _, ap := range rec.Approvals {
		rows = append(rows, []string{
			pipeline.SourceRecordID(sourceID, "approval", rec.SetID, ap.ID),
			rec.SetID,
			ap.ID,
			ap.Code,
			ap.Type,
			releaseFile,
			releaseFile,
		})
	}
	return rows
}

// IngredientRows builds the ingredient rows for one document, mirroring
// spl_xml._ingredient_rows.
func IngredientRows(rec DocumentRecord, sourceID, releaseFile string) [][]string {
	rows := make([][]string, 0, len(rec.Ingredients))
	for _, ing := range rec.Ingredients {
		rows = append(rows, []string{
			pipeline.SourceRecordID(sourceID, "ingredient", rec.SetID, ing.UNII, ing.Name),
			rec.SetID,
			ing.Name,
			ing.UNII,
			ing.Role,
			releaseFile,
			releaseFile,
		})
	}
	return rows
}

// SectionRows builds the per-section rows for one document, mirroring spl_xml._section_rows.
func SectionRows(rec DocumentRecord, sourceID, releaseFile string) [][]string {
	rows := make([][]string, 0, len(rec.Sections))
	for _, sec := range rec.Sections {
		rows = append(rows, []string{
			pipeline.SourceRecordID(sourceID, "section", rec.SetID, sec.LOINC),
			documentID(rec.SetID, sec.LOINC),
			rec.SetID,
			sec.LOINC,
			sec.Name,
			sec.Title,
			sec.RawText,
			sec.CleanText,
			releaseFile,
			releaseFile,
		})
	}
	return rows
}

// --- batch extraction ----------------------------------------------------------

// Tables holds the flattened normalized rows for a batch of input files, plus provenance
// (the input artifact ids and the accumulated parse-warning count).
type Tables struct {
	Documents   [][]string
	Sets        [][]string
	Approvals   [][]string
	Ingredients [][]string
	Sections    [][]string
	Warnings    int
	InputIDs    []string
}

// Extract parses the given SPL files and flattens them into the normalized tables. Row
// order is deterministic: files in `paths` order, documents in file order, sub-rows in
// document order. Files are parsed concurrently (bounded by limit; <=0 means unbounded)
// but results are reassembled in input order, so parallelism never affects output. The
// source artifact id for each file is its BLAKE3 file hash (matching the Python ref.blake3
// that underlies every source_record_id).
func Extract(ctx context.Context, paths []string, limit int) (*Tables, error) {
	type fileResult struct {
		sourceID    string
		releaseFile string
		docs        []DocumentRecord
		warnings    int
	}
	results := make([]fileResult, len(paths))
	g, gctx := errgroup.WithContext(ctx)
	if limit > 0 {
		g.SetLimit(limit)
	}
	for i, p := range paths {
		i, p := i, p
		g.Go(func() error {
			if err := gctx.Err(); err != nil {
				return err
			}
			sourceID, err := blake3store.HashFile(p)
			if err != nil {
				return fmt.Errorf("dailymed: hash %s: %w", p, err)
			}
			docs, err := ParseFile(p)
			if err != nil {
				return fmt.Errorf("dailymed: parse %s: %w", p, err)
			}
			w := 0
			for _, d := range docs {
				w += len(d.Warnings)
			}
			results[i] = fileResult{sourceID: sourceID, releaseFile: filepath.Base(p), docs: docs, warnings: w}
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}

	t := &Tables{}
	for _, r := range results {
		t.InputIDs = append(t.InputIDs, r.sourceID)
		t.Warnings += r.warnings
		for _, doc := range r.docs {
			t.Documents = append(t.Documents, DocumentRows(doc, r.releaseFile)...)
			t.Sets = append(t.Sets, SetRows(doc, r.sourceID, r.releaseFile)...)
			t.Approvals = append(t.Approvals, ApprovalRows(doc, r.sourceID, r.releaseFile)...)
			t.Ingredients = append(t.Ingredients, IngredientRows(doc, r.sourceID, r.releaseFile)...)
			t.Sections = append(t.Sections, SectionRows(doc, r.sourceID, r.releaseFile)...)
		}
	}
	return t, nil
}

// --- TSV output ----------------------------------------------------------------

// tsvField quotes a field for TSV exactly as polars write_csv(separator="\t") does: quote
// when the value is empty OR contains the separator, the quote char, or a line break
// (empty strings are emitted as "" so they stay distinct from a missing field); embedded
// quotes are doubled. Leading/trailing spaces alone do NOT trigger quoting. Verified
// against polars 1.43 byte-for-byte (see TestTSVFieldQuotingMatchesPolars).
func tsvField(s string) string {
	if s == "" || strings.ContainsAny(s, "\t\"\r\n") {
		return `"` + strings.ReplaceAll(s, `"`, `""`) + `"`
	}
	return s
}

func writeTSVLine(w io.Writer, fields []string) error {
	for i, f := range fields {
		if i > 0 {
			if _, err := io.WriteString(w, "\t"); err != nil {
				return err
			}
		}
		if _, err := io.WriteString(w, tsvField(f)); err != nil {
			return err
		}
	}
	_, err := io.WriteString(w, "\n")
	return err
}

// WriteTSV writes columns + rows as an uncompressed TSV with a header row and LF line
// endings (Tablassert-readable), matching the Python schemas.write_tsv output.
func WriteTSV(w io.Writer, columns []string, rows [][]string) error {
	bw := bufio.NewWriter(w)
	if err := writeTSVLine(bw, columns); err != nil {
		return err
	}
	for _, row := range rows {
		if err := writeTSVLine(bw, row); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// RenderTSV renders columns + rows to a TSV string (convenience for tests and callers).
func RenderTSV(columns []string, rows [][]string) string {
	var sb strings.Builder
	_ = WriteTSV(&sb, columns, rows)
	return sb.String()
}

// TableFile names one TSV output file with its columns and rows.
type TableFile struct {
	Name    string
	Columns []string
	Rows    [][]string
}

// TableFiles returns the five normalized tables as TSV file descriptors in a stable order
// (documents first — the locked public contract — then sets, approvals, ingredients,
// sections), mirroring the Python extractor's ref ordering.
func (t *Tables) TableFiles() []TableFile {
	return []TableFile{
		{"spl_documents.tsv", DocumentsColumns, t.Documents},
		{"spl_sets.tsv", SetsColumns, t.Sets},
		{"spl_approvals.tsv", ApprovalsColumns, t.Approvals},
		{"spl_ingredients.tsv", IngredientsColumns, t.Ingredients},
		{"spl_sections.tsv", SectionsColumns, t.Sections},
	}
}

// WriteDir writes every table as an uncompressed .tsv into dir (created if needed).
func (t *Tables) WriteDir(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	for _, tf := range t.TableFiles() {
		path := filepath.Join(dir, tf.Name)
		f, err := os.Create(path)
		if err != nil {
			return err
		}
		if err := WriteTSV(f, tf.Columns, tf.Rows); err != nil {
			f.Close()
			return err
		}
		if err := f.Close(); err != nil {
			return err
		}
	}
	return nil
}

// LooksLikeSPL reports whether a filename looks like an SPL input (.xml or .xml.gz),
// mirroring spl_xml._looks_like_spl.
func LooksLikeSPL(name string) bool {
	n := strings.ToLower(name)
	return strings.HasSuffix(n, ".xml.gz") || strings.HasSuffix(n, ".xml")
}

// ListSPLFiles returns the SPL files directly under dir (non-recursive), sorted by path
// for deterministic processing order. A single .xml/.xml.gz file path is returned as-is.
func ListSPLFiles(dir string) ([]string, error) {
	info, err := os.Stat(dir)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		if LooksLikeSPL(dir) {
			return []string{dir}, nil
		}
		return nil, fmt.Errorf("not an SPL file: %s", dir)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var paths []string
	for _, e := range entries {
		if e.IsDir() || !LooksLikeSPL(e.Name()) {
			continue
		}
		paths = append(paths, filepath.Join(dir, e.Name()))
	}
	sort.Strings(paths)
	return paths, nil
}

// --- low-level helpers ---------------------------------------------------------

// collapseWS collapses runs of whitespace to single spaces (mirrors the legacy Python
// " ".join(text.split()): strings.Fields splits on unicode whitespace runs, dropping
// empties, exactly like str.split()).
func collapseWS(s string) string { return strings.Join(strings.Fields(s), " ") }

// unii normalizes a UNII code to the "UNII:<code>" form ("" if absent), mirroring
// spl_xml._unii.
func unii(code string) string {
	if code == "" {
		return ""
	}
	return "UNII:" + code
}

// looksLoinc is a loose LOINC shape check: <digits>-<something> (mirrors spl_xml._looks_loinc).
func looksLoinc(code string) bool {
	dash := strings.Index(code, "-")
	if dash <= 0 {
		return false
	}
	for _, r := range code[:dash] {
		if r < '0' || r > '9' {
			return false
		}
	}
	return dash+1 < len(code)
}
