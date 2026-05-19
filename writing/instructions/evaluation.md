# evaluation.md
## Section Evaluation — Run After Each Section Draft

The agent must pass this check before moving to the next section.
Flag any failed check with [FAIL] and explain what needs fixing.
Only mark a section complete when all checks pass.

---

## Universal Checks (Apply to Every Section)

- [ ] Active voice used throughout (passive only in methodology procedures)
- [ ] No vague language: "high", "low", "better", "significant" without a number
- [ ] Every empirical claim has a specific number
- [ ] Every number has a table or figure reference (where applicable)
- [ ] Sentences are under 35 words
- [ ] Technical terms defined at first use
- [ ] Citation format matches JMR/AMA style: (Author Year) no comma
- [ ] No hedging on directly observed findings

---

## Section-Specific Checks

### Abstract
- [ ] 150 words or fewer
- [ ] No citations
- [ ] No jargon without definition
- [ ] Contains at least 3 specific numerical findings
- [ ] States the methodological contribution clearly
- [ ] States a practical implication
- [ ] 4–6 keywords provided

### Introduction
- [ ] Opens with business problem, not method
- [ ] Central claim stated by end of section
- [ ] 3–4 specific contributions listed
- [ ] Roadmap paragraph present
- [ ] No results reported
- [ ] 800–1000 words

### Literature Review
- [ ] Four subsections present
- [ ] Each subsection identifies a gap
- [ ] Each gap connects to a paper contribution
- [ ] Minimum 20 citations
- [ ] No one-by-one paper summaries — synthesis only
- [ ] 1000–1400 words

### Methodology
- [ ] Equations numbered (Eq 1–Eq 8 minimum)
- [ ] Every symbol defined
- [ ] Scenario table present
- [ ] Parameter table present (assumed vs estimated)
- [ ] Fixed-parameter design justified explicitly
- [ ] Replicability paragraph present (seed, script, data location)
- [ ] 1500–2000 words

### Results
- [ ] Each subsection leads with finding, not table description
- [ ] All 6 required tables present or referenced
- [ ] All 4 required figures present or referenced
- [ ] Channel attribution finding prominently included (not buried)
- [ ] ARDL S2→S4 reversal highlighted as cautionary finding
- [ ] MCMC S5 supplementary result included
- [ ] No interpretation — save for Discussion
- [ ] 2000–2500 words

### Discussion
- [ ] No new results introduced
- [ ] Three reviewer objections addressed
- [ ] Robustness taxonomy defined with tier boundaries
- [ ] Practical implication stated clearly
- [ ] Connected back to paper's central claim
- [ ] Limitations acknowledged honestly
- [ ] 800–1200 words

### Conclusion
- [ ] No new findings
- [ ] Three contributions stated in plain language
- [ ] Decision framework table or tree present
- [ ] Future research directions are specific (not generic)
- [ ] 400–600 words total

### References
- [ ] Every in-text citation has a reference entry
- [ ] Every reference entry is cited in text
- [ ] Format consistent throughout (JMR/AMA style)
- [ ] Minimum core references present (see references.md)
- [ ] Journal names italicised, article titles in quotes

---

## Content Accuracy Checks

Run these against paper_notes.md and experiment_log.csv:

- [ ] BSTS pause-window ratio cited as 1.02× (not approximated)
- [ ] MCMC S3 recovery cited as 99.0% (updated value, not 98.8%)
- [ ] ARDL S4 recovery cited as −19.8% (negative — sign flip)
- [ ] MCMC S5 supplementary recovery cited as 88.5%
- [ ] Framework averages S1–S4: F3 78.4%, F2 42.8%, F1 22.4%
- [ ] ARDL S2 channel attribution: 68.8% aggregate, 0% per channel
- [ ] Koyck S2 channel inversion: Social 59.3% > TV 2.2%
- [ ] Video LTC: 0% for all non-MCMC models in S3/S4/S5

---

## Output Format for Evaluation

After running checks, output:

```
SECTION: [name]
STATUS: COMPLETE / NEEDS REVISION

PASSED: [N] / [total] checks
FAILED CHECKS:
  - [check name]: [what needs fixing]

READY TO PROCEED: YES / NO
```

Do not proceed to the next section until STATUS = COMPLETE.
