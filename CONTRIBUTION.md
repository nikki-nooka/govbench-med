# Research Contribution Statement

## One-Line Claim
We introduce GovBench-Med, the first benchmark that quantifies the safety-per-cost
Pareto frontier of governance mechanisms in multi-agent LLM clinical diagnosis systems.

---

## The Problem Nobody Has Measured
Every paper in multi-agent medical AI either:
- Maximizes accuracy and ignores what safety overhead costs, OR
- Adds safety mechanisms and reports accuracy gains, but never maps the cost curve

Nobody has asked: *"If I add one more governance layer, exactly how much safety do I
gain, and what do I pay for it?"*

That is precisely what this paper measures.

---

## Three Novel Claims (what reviewers will evaluate you on)

### Claim 1: Governance-as-a-dial
We operationalize AI governance as a 5-level taxonomy (G0–G4) that is:
- Mechanically distinct at each level (not just "more prompting")
- Applicable to any multi-agent LLM framework
- Reproducible across models and datasets

**Why this is new**: Prior work treats governance as binary (guardrails on/off).
We treat it as a continuous spectrum with measurable granularity.

### Claim 2: Safety is not accuracy
We introduce a *Clinical Safety Score (CSS)* that separates:
- Critical Miss Rate (CMR): missing a life-threatening diagnosis
- Hallucination Impact Rate (HIR): fabricated findings that change the output
- Unsafe Reassurance Rate (URR): overconfident wrong diagnoses

This is distinct from accuracy. A model can be 85% accurate and still have
a 30% critical miss rate on the 15% of cases that matter most.

**Why this is new**: All cost-tradeoff papers use accuracy as the quality axis.
We use harm-weighted safety — a much more clinically meaningful signal.

### Claim 3: Governance Efficiency (GE) as a deployment metric
We define:
    GE = ΔCSS / ΔCost

where Cost = normalized(tokens + latency).

GE tells a hospital exactly: "for every $1 of extra compute you spend on governance,
you gain X% in clinical safety." This is a deployable decision metric.

**Why this is new**: No existing paper provides a single metric for governance ROI.

---

## How We Differ From the Three Closest Papers

| Paper | What they do | Our difference |
|-------|-------------|----------------|
| TeamMedAgents (2025) | Pareto frontier of accuracy vs. token cost | We use *safety* (not accuracy) as the Y-axis, and *governance level* as the X variable |
| ConfAgents (2025) | Adaptive cost-accuracy tradeoff | They adapt based on case difficulty; we vary governance mechanisms systematically |
| MedSafe-Dx (2026) | Safety benchmark for single-agent LLMs | Single-agent only; no multi-agent governance, no cost measurement |

---

## Abstract (Draft v1)

Multi-agent LLM systems are increasingly proposed for clinical diagnosis, yet the cost
of making them safe remains unquantified. We introduce GovBench-Med, a reproducible
benchmark that evaluates five governance levels — from bare multi-agent inference to
full consensus-with-verification — across three open-source language models on 300
clinical cases drawn from MedQA-USMLE and DDXPlus. For each configuration, we measure
Clinical Safety Score (CSS), token cost, and latency, then chart the governance–cost
Pareto frontier. We find that governance follows a law of diminishing returns: moving
from G0 to G2 yields 73% of attainable safety gains at 31% of G4's cost, while G3→G4
buys marginal safety at 2× overhead. We introduce Governance Efficiency (GE = ΔCSS/ΔCost)
as a deployment metric and show that it varies significantly across model families,
with open-source models achieving competitive GE to proprietary systems at zero licensing
cost. GovBench-Med provides the first systematic, reproducible characterization of the
governance–cost tradeoff, offering concrete guidance for safe and cost-effective
deployment of multi-agent clinical AI.

*[Note: numbers above are placeholders — replace with actual experimental results]*

---

## IEEE Access Framing
- **Scope**: Biomedical informatics + AI systems evaluation
- **Contribution type**: Benchmark + empirical study
- **Reproducibility**: Full open-source, no proprietary APIs required
- **Impact statement**: Provides deployment guidance for healthcare AI governance
