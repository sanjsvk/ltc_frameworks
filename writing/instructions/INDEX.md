# Writing Skills — LTC Framework Research Paper
## Agent: Start here. Read this file first, then follow section references.

---

## What This Folder Contains

This folder contains all writing guidance for drafting the LTC Framework paper.
Each file handles one specific job. Do not try to hold all guidance in memory at once —
load only the file you need for the task at hand.

---

## File Map

```
skills/
├── INDEX.md                  ← You are here. Start every session here.
├── writing_skill.md          ← Global writing standards. Read before writing any section.
├── sections/
│   ├── abstract.md           ← Abstract guidance
│   ├── introduction.md       ← Introduction guidance
│   ├── literature.md         ← Literature review guidance
│   ├── methodology.md        ← Methodology guidance
│   ├── results.md            ← Results guidance
│   ├── discussion.md         ← Discussion guidance
│   ├── conclusion.md         ← Conclusion + recommendations guidance
│   └── references.md         ← Citation and reference formatting
├── visuals.md                ← Tables and figures guidance
└── evaluation.md             ← Quality check — run after each section is drafted
```

---

## How To Use This Skill Folder

### Starting a new session
1. Read this file (INDEX.md)
2. Read @writing_skill.md — applies to every section, every time
3. Read @evaluation.md — know the quality bar before you start writing
4. Load paper_notes.md and experiment_log.csv for findings and metrics

### Writing a section
1. Read the relevant section file from @sections/
2. Read @visuals.md if the section includes tables or figures
3. Draft the section following the file's guidance
4. Run the evaluation check from @evaluation.md before moving on

### Section order
Write in this order. Each section informs the next.
```
1. Methodology     → establishes what was done
2. Results         → reports what was found
3. Discussion      → interprets findings
4. Introduction    → frames the paper around known findings
5. Literature      → positioned against known findings
6. Abstract        → summarises the complete paper
7. Conclusion      → closes the arc
8. References      → compiled last
```

---

## Paper Identity

**Title (working):** Estimating Long-Term Media Contributions in Marketing Mix Models:
A Reproducible Benchmarking Framework

**Target journals (in order of preference):**
- Journal of Marketing Research (JMR) — primary
- Marketing Science — secondary
- International Journal of Research in Marketing (IJRM) — tertiary

**Paper type:** Quantitative methodology paper with empirical benchmarking
**Core claim:** Static adstock methods systematically fail to recover LTC from sustained
brand investment. A latent stock state-space formulation is structurally robust
where adstock-based methods fail.

**Key constraint:** Marketing Science requires replication code and data on acceptance.
JMR requires research transparency materials. Build all writing with this in mind —
the paper must be fully reproducible.

---

## Source Files (always available)
- `paper_notes.md` — 16 logged findings with evidence and paper language
- `experiment_log.csv` — all metrics across 10 models × 5 scenarios
- `S3_S4_S5_CHANNEL_ATTRIBUTION.txt` — channel-level validation results
- `mmm_synthetic_generator.py` — data generation code (for methodology section)
- `PROJECT_BRIEF.md` — full experiment design context
