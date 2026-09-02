# Regulatory Framework Discussion Outline

This is the final section of the paper (Section 7 or 8 depending on structure).
It connects our empirical findings to emerging AI governance regulations.

---

## 7. Discussion: Implications for Agentic Clinical AI Regulation

### 7.1 The Governance Calibration Problem

Current regulatory approaches (FDA GMLP, EU AI Act, NIST AI RMF) treat safety
governance as a binary compliance checkbox: either you have guardrails or you don't.
Our empirical findings challenge this framing by demonstrating that governance follows
a law of diminishing returns — each additional oversight mechanism contributes
progressively less to clinical safety while consuming more computational resources.

**Key finding to cite**: If GE drops below 1 at G3, then mandating G4-level governance
for all clinical AI deployments may be miscalibrated — consuming 2× the compute for
marginal safety gains, while the same resources could be redirected to expanding access
to a well-governed G2 system to more patients.

### 7.2 Risk-Proportional Governance

Our benchmark supports a **risk-proportional governance** model where:

- **Low-acuity routine cases**: G1–G2 sufficient (verifier catches hallucinations, roles
  prevent format errors). Cost-effective at scale.
- **Medium-acuity ambiguous cases**: G3 needed (consensus prevents unsafe reassurance).
  The abstention mechanism is clinically appropriate — some cases genuinely warrant
  human escalation.
- **High-acuity life-threatening cases**: G4 justified despite cost. The ethics critic
  and audit trail are proportionate to the severity of potential harm.

This maps onto the EU AI Act's risk-based classification:
- **Minimal risk** (routine triage): G1–G2
- **Limited risk** (complex diagnosis): G3
- **High risk** (life-threatening): G4

### 7.3 Quantified Compliance Evidence

Our Governance Efficiency (GE) metric provides regulators with a concrete,
auditable measure of governance adequacy. Instead of asking "do you have
safety guardrails?" regulators could ask:

> "What is your system's GE at the deployed governance level, and does it exceed
> the minimum threshold for the clinical risk category?"

This transforms governance from a qualitative design choice into a quantitative
compliance requirement — similar to how clinical trials require statistical
significance thresholds, not just "did you run a trial?"

### 7.4 The Cost of Over-Governance

A counterintuitive finding: overly governed systems can be clinically harmful
because:

1. **Abstention fatigue**: If G3 abstains on 40% of cases, clinicians stop
   trusting the system and revert to manual workflows — negating the AI's benefit.
2. **Latency penalties**: In time-critical scenarios (sepsis, stroke), a 7×
   governance overhead could delay diagnosis beyond the treatment window.
3. **Resource misallocation**: Compute spent on governance for routine cases
   could fund deployment to underserved clinics.

Regulatory frameworks should explicitly account for the cost of
over-governance, not just under-governance.

### 7.5 Recommendations for Regulators

Based on our empirical governance-cost curves, we propose:

1. **Mandate governance benchmarks, not just governance mechanisms.**
   Require developers to publish governance-cost curves (like our Pareto frontier)
   showing what oversight level achieves what safety level at what cost.

2. **Adopt risk-proportional governance tiers.**
   Match governance requirements to clinical risk: G2 for routine, G3 for
   complex, G4 for critical. Avoid one-size-fits-all mandates.

3. **Require open-source measurement harnesses.**
   Regulators should mandate that governance claims are reproducible using
   open benchmarks (like GovBench-Med), not just self-reported metrics.

4. **Establish minimum GE thresholds.**
   If GE < 0.5 at a governance level, the system is over-governed for that
   clinical context. This prevents wasteful over-provisioning.

5. **Audit trail as compliance evidence.**
   The structured audit logs from G4 provide a machine-readable compliance
   record that regulators can inspect programmatically.

### 7.6 Limitations of This Discussion

- Our findings are based on synthetic/simulated clinical cases, not real patient data
- Regulatory implications are interpretive, not legally binding
- Different healthcare systems may have different risk tolerance thresholds
- The cost model (tokens + latency) does not include human reviewer time

---

## How This Section Connects to the Paper

```
Section 1 (Intro):     "nobody has measured governance cost"
Section 2 (Related):   "prior work ignores the cost axis"
Section 3 (Method):    "here's our governance taxonomy + metrics"
Section 4 (Setup):     "here's how we run experiments"
Section 5 (Results):   "here's the Pareto frontier + GE curves"
Section 6 (Ablation):  "here's what each component contributes"
Section 7 (Discussion): "here's what this means for regulation"  ← THIS
Section 8 (Conclusion): "first benchmark, open-source, call to action"
```

The regulatory discussion elevates the paper from "benchmark study" to
"policy-relevant research" — which is exactly what IEEE Access reviewers
look for in biomedical AI papers.
