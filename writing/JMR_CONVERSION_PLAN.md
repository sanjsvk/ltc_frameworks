# JMR Format Conversion Plan — 10 Steps

**Current State:** Markdown document (~14,500 words with structure), 16 figures, 14 references  
**Target:** Journal of Marketing Research submission-ready document (≤50 pages, AMA formatted)

---

## Phase 1: Format Decision (Step 1-2)

### Step 1: Format Selection — Recommendation: **Microsoft Word (.docx)**

**Why Word over LaTeX:**
- ✅ JMR explicitly accepts Word (via Manuscript Central)
- ✅ Reviewers typically use Word for commenting (Track Changes)
- ✅ Table formatting easier in Word
- ✅ Figure placement control simpler
- ✅ Reference management (Zotero/Mendeley) integrates better with Word
- ⚠️ LaTeX: Not required by JMR; adds complexity for collaborative review

**Decision:** Convert Markdown → Word (.docx) using Pandoc or manual conversion

---

### Step 2: Word Count Audit & Consolidation

**Current breakdown (approximate):**
- Sections 1-10 (core content): ~9,450 words
- References: ~800 words
- Table captions + figure captions: ~1,500 words
- Total: ~11,750 words (WELL under 50-page limit)

**Page estimate:** ~25-30 pages (including 16 figures at ~1 page per 2-3 figures)

**Action:** NO major cuts needed. Target is 9,500–10,500 words (leaves room for formatting, spacing, figure placement).

---

## Phase 2: Content Restructuring (Step 3-5)

### Step 3: Verify Section Structure Matches JMR Standards

**Current structure (Sections 1-10):**
1. ✅ Abstract (200 words, unstructured) — PASS
2. ✅ Introduction + Literature Review (1,280 words) — PASS
3. ✅ Methodology (1,847 words) — PASS
4-8. ✅ Results (5 sections: Framework, Scenario, Channel, Calibration, Anomaly) — PASS
9. ✅ Discussion (1,166 words) — PASS
10. ✅ Recommendations (522 words) — Consider merging with Discussion or adding as "Implications"

**Recommended restructure:**
- Abstract (as is)
- Introduction + Literature Review (combine into single section "1. Introduction")
- Methodology (as is)
- Results (merge Sections 4-8 into "3. Results" with subsections A-E)
- Discussion (merge Section 9 + 10 into "4. Discussion & Implications")
- References

**Rationale:** JMR prefers 4-5 major sections (not 10) to reduce visual fragmentation.

---

### Step 4: Consolidate Lengthy Sections

**Sections to compress:**

| Section | Current | Target | Action |
|---------|---------|--------|--------|
| Intro + Lit Rev | 1,280 | 1,100 | Remove 2-3 citations; consolidate lit review summary |
| Methodology | 1,847 | 1,600 | Remove repetitive DGP description; move detailed equations to footnotes |
| Results | 5,000+ | 4,200 | Remove redundant model descriptions; consolidate S3-S5 into bullet summaries |
| Discussion | 1,166 | 900 | Merge recommendation findings into Discussion conclusion |

**Estimated savings:** 500-600 words → new total ~9,500 words (still comfortable for 25-page document)

---

### Step 5: Remove/Relocate Web Appendices

**Current appendices (in master document):**
- Detailed channel-level metrics (Section 6 tables)
- Robustness score calculations (Section 7)
- Anomaly diagnostic details (Section 8)

**Recommendation:**
- Keep critical tables (channel rankings, robustness scores) IN main document
- Move detailed diagnostic tables → Web Appendices (not in 50-page count)
- Move replication data/code references → Supplementary Materials section

**Web Appendix structure:**
- A. Extended channel-level attribution (all 10 models × 5 scenarios)
- B. Scenario-specific robustness metrics (pause ratios, S4 break analysis)
- C. Model parameter specifications (hyperparameter grids)
- D. Replication package guide (code, data, README)

---

## Phase 3: Formatting & Styling (Step 6-8)

### Step 6: Convert to Word Format

**Tools & process:**

**Option A (Recommended): Manual Word template**
1. Create blank Word document with JMR AMA template
2. Copy-paste from markdown, reformatting as you go
3. Keep markdown file as reference
4. **Effort:** 2-3 hours | **Quality:** High (full control)

**Option B: Pandoc conversion**
```bash
pandoc MASTER_DOCUMENT_FINAL.md -o LTC_Frameworks_JMR.docx \
  --reference-doc=jmr-template.docx \
  --csl=ama.csl
```
- **Effort:** 30 minutes (if template exists) | **Quality:** Medium (manual cleanup needed)

**Option C: Hybrid (Recommended)**
1. Export markdown → docx via Pandoc
2. Manually verify formatting, references, figures
3. Apply AMA style tweaks

---

### Step 7: Apply AMA Reference Formatting

**Current state:** 14 references in AMA style (verified in Phase 2 validation)

**Word implementation:**
- ✅ Use Zotero or Mendeley with AMA style profile
- ✅ Insert citations as "Fields" in Word (auto-update capability)
- ✅ Generate bibliography from Zotero/Mendeley
- ✅ Manually verify alphabetical order, DOIs, title case

**Expected output:** References section (1 page) with all 14 citations properly formatted

---

### Step 8: Figure & Table Formatting

**16 Figures — Requirements:**
- ✅ Resolution: 300 dpi (current PNG files)
- ✅ Placement: Embed in Word document (don't use external files)
- ✅ Captions: Under each figure in JMR style: "**Figure X: Title.** *Description.*"
- ✅ Cross-references: Use Word "Caption" feature for auto-numbering
- ✅ Size: ~3.5–4 inches wide (fits single column)

**5 Tables — Requirements:**
- ✅ AMA style: No vertical lines, minimal horizontal rules
- ✅ Titles: Above table, bold, numbered: "**Table 1. Description.**"
- ✅ Notes: Below table in smaller font (e.g., "Note: Recovery % = accuracy in recovering true LTC")
- ✅ Word formatting: Use table styles; ensure consistent fonts (11pt)

**Action checklist:**
- [ ] Embed all 16 PNG figures in Word
- [ ] Verify DPI (open each in image viewer; check properties)
- [ ] Apply consistent captions using Word Caption feature
- [ ] Format all 5 tables with AMA style (no vertical lines)
- [ ] Verify table/figure numbering auto-updates

---

## Phase 4: Compliance & Submission (Step 9-10)

### Step 9: Apply JMR Compliance Checklist

**30-item checklist from JMR_SUBMISSION_GUIDELINES.md:**

**Content (8 items):**
- [ ] Abstract ≤ 200 words, unstructured
- [ ] 3+ keywords provided
- [ ] No hedging language (avoid "may," "might," "could")
- [ ] All claims supported by data/references
- [ ] Novel contribution stated in Intro/Discussion
- [ ] Limitations discussed
- [ ] Implications for practitioners included
- [ ] All figures/tables serve clear purpose

**Formatting (8 items):**
- [ ] 12pt font, double-spaced body text
- [ ] References single-spaced (AMA style)
- [ ] Margins 1 inch all sides
- [ ] Page numbers (header/footer)
- [ ] Running title (optional, but recommended)
- [ ] All footnotes converted to endnotes or integrated into text
- [ ] No tracking changes or comments in final version
- [ ] File saved as .docx (not .doc or .pdf)

**Figures & Tables (6 items):**
- [ ] All 16 figures included and numbered sequentially
- [ ] All figure captions complete and descriptive
- [ ] All 5 tables formatted consistently
- [ ] No figures/tables in color (unless you pay for color printing)
- [ ] PNG resolution ≥ 300 dpi verified
- [ ] All cross-references in text (e.g., "see Figure 5")

**Citations & References (5 items):**
- [ ] In-text citations formatted (LastName Year) or LastName (Year)
- [ ] All in-text citations have corresponding reference entries
- [ ] All references in alphabetical order by first author
- [ ] 14 references verified through AMA style guide
- [ ] DOIs included where available

**Data & Reproducibility (3 items):**
- [ ] Replication data prepared (in CSV/Excel format)
- [ ] README file explains data structure
- [ ] Code provided (R/Python scripts with comments)

---

### Step 10: Final Proofread & Manuscript Central Upload

**Final checks before submission (1-2 hours):**

1. **Spelling & grammar** (use Word spell-check, then Grammarly)
2. **Consistency audit:**
   - Framework terminology (F1, F2, F3 vs Framework 1, 2, 3)
   - Model names (BSTS, Kalman DLM, MCMC — verify caps throughout)
   - Scenario labels (S1-S5 vs Scenario 1-5)
   - Metric names (recovery_accuracy vs recovery accuracy)
3. **Cross-reference validation:**
   - All 8 "see Section X" parentheticals working
   - All figure/table numbers accurate
4. **Test PDF conversion:** Save as PDF to verify rendering (required by some journals)

**Manuscript Central submission:**
- Prepare required files:
  - [ ] Main manuscript (LTC_Frameworks_JMR.docx)
  - [ ] Title page (separate from main document)
  - [ ] Figure files (if required as separate uploads)
  - [ ] Replication package (upload as supplementary material)
  - [ ] Cover letter (3-5 paragraphs)

---

## Timeline & Effort Estimate

| Phase | Steps | Effort | Timeline |
|-------|-------|--------|----------|
| **Format Decision** | 1-2 | 30 min | Day 1 (morning) |
| **Word Count & Restructure** | 3-5 | 2-3 hrs | Day 1 (afternoon) |
| **Formatting & Styling** | 6-8 | 2-3 hrs | Day 2 (morning) |
| **Compliance & Submission** | 9-10 | 1-2 hrs | Day 2 (afternoon) |
| **TOTAL** | **10 steps** | **6-9 hours** | **2 days** |

---

## Critical Decision Points

1. **Word vs LaTeX:** Recommend Word (simpler, JMR standard)
2. **Section consolidation:** Merge into 4-5 major sections (vs current 10)
3. **Web appendices:** Move extended tables to supplementary materials
4. **Figure color:** Keep grayscale (save author cost)
5. **Replication package:** Include code (recommended for reproducibility score)

---

## Success Criteria

After completion, document should:
- ✅ Be ≤ 50 pages (estimated 25-30 pages)
- ✅ Comply with all JMR formatting requirements
- ✅ Have all 16 figures properly embedded and captioned
- ✅ Have all 14 references in AMA format
- ✅ Pass the 30-item compliance checklist
- ✅ Be ready to upload to Manuscript Central

---

**Recommendation:** Start with Step 6 (Word template creation). If you'd like, I can assist with any of the conversion steps.
