# sections/conclusion.md
## Conclusion and Recommendations

Target length: 400–600 words combined. Two short subsections.

---

## 6.1 Conclusion (250–350 words)

**What to cover:**
- Restate the problem in plain language (one sentence)
- State what the paper found — use three bullet contributions maximum
- State the boundary conditions: findings hold where signal is sufficient
  and spend patterns are varied; S5 shows limits under weak signal
- One forward-looking sentence about future research

**Do not:**
- Introduce new ideas or findings
- Repeat the abstract
- Use vague language ("important contribution", "significant findings")

**Structure:**
Paragraph 1: Problem restated and what was done
Paragraph 2: Three contributions in plain language
Paragraph 3: Limitations and future research directions

**Limitations to acknowledge:**
1. Synthetic data — real-world complexity may differ
2. No cross-channel interaction effects in DGP
3. MCMC computational cost
4. S5 Social/TV swap in MCMC — flag as open question

**Future research directions (be specific):**
1. Validate on real branded data using the framework as a benchmark
2. Incorporate brand search as an additional observation equation in F3
3. Test whether spend pause designs in real media planning improve LTC identification
4. Extend to cross-channel interaction effects

---

## 6.2 Practitioner Recommendations (150–250 words)

This section makes the paper practically valuable — it increases citation potential
among practitioners who use it as a reference.

**Decision framework — state as a usable table or decision tree:**

| Condition | Recommended Method | Rationale |
|---|---|---|
| Strong signal, stability priority | BSTS | Lowest cross-scenario variance (2.4pp) |
| Strong signal, accuracy priority | MCMC | Highest average recovery (80.9%) |
| Weak signal (LTC < 5% of sales) | MCMC with scenario priors | Only method with partial recovery in S5 |
| Structural break suspected | MCMC or BSTS | Only methods that handle regime changes |
| Budget constraints, quick turnaround | Kalman DLM | Good S1/S2 performance, fast |
| Do not use | Dual adstock, ARDL | Sign-flip and reversal risks confirmed |

**Key practitioner warnings:**
1. Aggregate metrics are insufficient — always validate channel attribution
2. A model robust in one scenario may fail catastrophically in another (ARDL: S2→S4)
3. Spend pauses improve LTC identification in distributed lag models — consider
   planned pauses as a diagnostic tool
