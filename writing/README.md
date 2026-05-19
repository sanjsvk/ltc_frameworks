# Paper Writing Project Structure

## Purpose
Organize LTC Frameworks research paper drafting into three clear stages:
1. **Instructions** — guidance, journal standards, section-specific rules
2. **Drafts** — actual section content as written
3. **Evaluation** — compliance verification before finalizing

---

## Folder Guide

### `instructions/`
Contains all guidance, standards, and section-specific writing rules.

**Start here before drafting any section:**
- `INDEX.md` — Overview of skills framework and recommended workflow
- `writing_skill.md` — Global writing standards (active voice, precision, hedging, citations)
- `PHASE1_SETUP_WRITEUP.md` — Comprehensive reference covering all journal requirements, section-by-section guidance, evaluation checklists
- `methodology.md` — Specific guidance for Methodology section (equations, tables, replicability)
- `introduction.md` — Specific guidance for Introduction section
- `results.md` — Specific guidance for Results sections
- `discussion.md` — Specific guidance for Discussion section
- `evaluation.md` — Quality checklist for all sections
- `visuals.md` — Tables and figures formatting standards
- `references.md` — Citation and reference formatting

### `drafts/`
Contains actual section drafts as they are written.

**Current status:**
- ✅ `INTRODUCTION_DRAFT.md` — Section 2, 930 words, all evaluations passed

**Planned:**
- `METHODOLOGY_DRAFT.md` — Section 3 (next)
- `RESULTS_SECTION_4_DRAFT.md` — Framework Comparison
- `RESULTS_SECTION_5_DRAFT.md` — Scenario Sensitivity
- `RESULTS_SECTION_6_DRAFT.md` — Channel Attribution
- `RESULTS_SECTION_7_DRAFT.md` — Calibration Sensitivity
- `DISCUSSION_DRAFT.md` — Section 9
- `RECOMMENDATIONS_DRAFT.md` — Section 10
- `ABSTRACT_DRAFT.md` — Section 1 (written last)
- `CONCLUSION_DRAFT.md` — Section 11
- `REFERENCES_DRAFT.md` (compiled last)

### `evaluation/`
Contains evaluation reports after each section is drafted.

**Current status:**
- ✅ `INTRODUCTION_EVALUATION.md` — 16-point assessment, all checks passed

**Workflow:**
1. Draft a section in `drafts/`
2. Run evaluation against checklist in `instructions/evaluation.md`
3. Save assessment report in `evaluation/`
4. Revise draft until all checks pass
5. Commit both draft and evaluation together

---

## Writing Workflow

For each section:

1. **Read the guidance**
   - Check `instructions/INDEX.md` for that section's guidance file
   - Read `instructions/writing_skill.md` (applies to all sections)
   - Read section-specific file (e.g., `instructions/methodology.md`)

2. **Draft the section**
   - Create file in `drafts/` following the guidance
   - Reference `paper_notes.md` and `comprehensive_analysis/` for findings and data
   - Ensure compliance with journal standards while drafting

3. **Evaluate**
   - Use checklist from `instructions/evaluation.md`
   - Save assessment report in `evaluation/` with same name as draft
   - Note any checks that failed and required revisions

4. **Revise and finalize**
   - Make revisions to draft
   - Re-evaluate until all checks pass
   - Commit when complete

---

## Recommended Writing Order

Per `instructions/INDEX.md` (differs from final paper order):

1. **Methodology** (Section 3) — Establishes what was done; foundation for Results
2. **Results** (Sections 4-7) — Reports what was found; informs Discussion & Introduction
3. **Discussion** (Section 9) — Interprets findings; shapes Introduction positioning
4. **Introduction** (Section 2) — Frames paper around known findings
5. **Literature** (Section 8) — Positioned against known findings
6. **Abstract** (Section 1) — Summarizes complete paper **LAST**
7. **Conclusion** (Section 11) — Closes the arc
8. **References** — Compiled at end

---

## Key Resources Referenced in Drafts

**Findings and Data:**
- `/c/github/ltc/paper_notes.md` — 16 validated research findings with evidence
- `/c/github/ltc/comprehensive_analysis/` — S1-S5 scenario results, channel attribution, optimization results
- `/c/github/ltc/CLAUDE.md` — Project context, frameworks, experimental results
- `/c/github/ltc/RUN_LOG.md` — Detailed experiment logs, convergence diagnostics

**Citation Locations:**
- Clarke, Darral G. (1976) — Foundational adstock paper [VERIFY: Journal of Marketing Research]
- Nerlove & Arrow (1962) — Foundational LTC/stock model [VERIFY]
- Additional MMM benchmark citations — [TO IDENTIFY for Paragraph 3]

---

## Current Status

| Section | Draft | Evaluation | Status |
|---------|-------|------------|--------|
| Abstract (1) | ⏳ | ⏳ | Written LAST |
| Introduction (2) | ✅ | ✅ | Complete, citations pending verification |
| Methodology (3) | ⏳ | ⏳ | Next to draft |
| Results A (4) | ⏳ | ⏳ | Pending Methodology |
| Results B (5) | ⏳ | ⏳ | Pending Methodology |
| Results C (6) | ⏳ | ⏳ | Pending Methodology |
| Results D (7) | ⏳ | ⏳ | Pending Methodology |
| Discussion (9) | ⏳ | ⏳ | Pending Results |
| Recommendations (10) | ⏳ | ⏳ | Pending Discussion |
| Conclusion (11) | ⏳ | ⏳ | Pending Recommendations |
| References | ⏳ | ⏳ | Compiled at end |

---

## Next Steps

1. ✅ Create folder structure (DONE)
2. ⏳ Verify citations in Introduction (Paragraph 2: Clarke 1976, Nerlove & Arrow 1962; Paragraph 3: additional MMM benchmark papers)
3. ⏳ Draft Methodology Section 3
4. ⏳ Continue with Results sections
