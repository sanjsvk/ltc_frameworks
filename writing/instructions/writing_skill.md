# writing_skill.md
## Global Writing Standards — Apply to Every Section

Read this before writing anything. These rules apply throughout the entire paper.

---

## Voice and Tone

**Active voice by default.**
- Write: "MCMC recovers 88.5% of true LTC under weak signal conditions"
- Not: "88.5% recovery was achieved by MCMC under weak signal conditions"

**Passive voice is acceptable only in methodology** when describing procedures.
- Acceptable: "Parameters were fixed to their true data-generating values"

**Tone: precise, confident, readable.**
JMR explicitly states: "Write in an interesting, readable manner. The journal is
designed to be read, not deciphered." Marketing Science emphasises readability
to as broad an audience as feasible. IJRM values clarity in factual reporting.

---

## Claim Standards — Every Empirical Claim Follows This Pattern

```
[Finding] + [specific number] + [table/figure reference] + [mechanism]
```

Example:
"BSTS achieves a pause-window robustness ratio of 1.02 (Table 4), maintaining
nearly identical error distribution across smooth and discontinuous spend regimes —
a result explained by its latent stock transition equation, which depreciates
independently of the spend observation."

**Never write:**
- "high recovery" → write "82.4% recovery"
- "performs better" → write "outperforms by 13.2 percentage points"
- "the model fails" → write "returns 0% LTC recovery"
- "as expected" → cite the theoretical prediction that generated the expectation
- "interestingly" → let the number speak

---

## Precision Rules

- All percentages to one decimal place: 82.4%, not 82% or 82.43%
- All ratios to two decimal places: 1.02×, not 1.0× or 1.021×
- Always specify scenario when citing a metric: "(S3, Table 3)"
- Always specify model when citing a result: "MCMC achieves..."
- Dollar figures: $M with one decimal: $0.49M, not $490K or $0.492M

---

## Hedging — When and How

**Hedge when** mechanism is inferred, not directly observed:
- "suggests", "is consistent with", "indicates", "may reflect"

**Do not hedge when** the finding comes directly from ground truth comparison:
- "MCMC recovers", "ARDL returns 0%", "the ratio is 1.02"

**Never hedge in the abstract.** State findings as facts.

---

## Sentence and Paragraph Structure

- Keep sentences under 35 words. Long sentences lose reviewers.
- One idea per paragraph. First sentence states the idea. Last sentence reinforces it.
- Use transitional phrases to connect paragraphs: "Building on this finding...",
  "This result is consistent with...", "In contrast..."
- Define every technical term at first use. Never assume the reader knows MMM jargon.

---

## What to Avoid

- Do not use "significant" to mean "large" — it has a statistical meaning
- Do not use "robust" loosely — define what robust means in context
- Avoid footnotes where possible — AMA journals prefer inline explanation
- Avoid acronyms in the abstract
- Do not use "novel" to describe your own work — let the contribution speak
- Do not say "to the best of our knowledge" — just make the claim and cite the gap

---

## Citation Format — JMR / AMA Style

**In-text:** Author last name and year, no comma between them.
- One author: (Clarke 1976)
- Two authors: (Srinivasan and Hanssens 2009)
- Three or more: (Jin et al. 2017)
- Multiple citations: (Clarke 1976; Hanssens et al. 2001)

**Reference list format:**
Author Last, First (Year), "Title of Article," *Journal Name*, Volume (Issue), pages.

Example:
Clarke, Darral G. (1976), "Econometric Measurement of the Duration of Advertising
Effect on Sales," *Journal of Marketing Research*, 13 (4), 345–357.

**AMA journals now require:** Exact p-values (not thresholds), standard errors in
tables, and effect sizes. Report all three where statistical tests are used.

---

## Equations

- Number every equation referenced in the text: (1), (2), etc.
- Define every symbol at first appearance
- State units explicitly in the surrounding text
- Follow each equation with one sentence of plain-language interpretation
- Centre equations on their own line

---

## Replicability (Required by All Three Target Journals)

Marketing Science requires code and data on acceptance.
JMR requires research transparency materials.
IJRM encourages data sharing.

Write all methods with this in mind:
- Every parameter must be stated explicitly
- Every analytical decision must be justified
- The synthetic data generator is referenced by name and seed value
- All results must be traceable to experiment_log.csv
