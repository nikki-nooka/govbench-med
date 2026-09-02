# Research Contribution Statement

## One-Line Claim
We introduce GovBench-Med, the first open-source benchmark that quantifies the
marginal cost-benefit of each governance layer in multi-agent clinical diagnosis
pipelines, enabling healthcare organizations to calibrate safety infrastructure
to their actual risk tolerance.

---

## The Problem This Paper Solves
Healthcare organizations deploying multi-agent clinical AI face a governance
dilemma:

- **Over-provision safety**: waste compute, slow time-critical workflows,
  increase cost per diagnosis
- **Under-provision safety**: risk hallucinated findings, unsafe recommendations,
  missed critical diagnoses

There is no empirical benchmark that tells you: "For THIS clinical risk level,
THIS governance configuration costs THIS much and delivers THIS much safety."

That is precisely what this paper measures.

---

## What the Abstract Promises (and How We Deliver It)

### Promise 1: Multi-agent pipeline variants with progressively increasing oversight
**Deliverable**: Governance levels G0–G4 (cumulative) + ablation variants
(BASE, +VER, +HITL, +COMB) that isolate each component.

| Level | What It Adds | Mechanism |
|-------|-------------|-----------|
| G0 (BASE) | Ungoverned baseline | 2 diagnostic agents, majority vote |
| G1 | Structured roles + guardrails | Role specialization + output validation |
| G2 (+VER) | Automated verifier agent | Cross-checks findings against case notes |
| G3 (+HITL) | Simulated HITL approval gate | Confidence-gated consensus + abstention |
| G4 (+COMB) | Combined governance layers | Ethics critic + audit trail + suppression |

### Promise 2: Research agent + report-writing agent baseline
**Deliverable**: G0 baseline with a diagnostician agent + report writer agent
that produces structured clinical reports.

### Promise 3: Rubric-based and LLM-as-judge scoring
**Deliverable**: 8-dimension clinical rubric evaluated by LLM judge:
- Completeness, Clinical Accuracy, Appropriate Uncertainty,
  Evidence Grounding, Safety, Structured Format, No Hallucination,
  Actionability (each scored 0–5)

### Promise 4: Symptom-based differential diagnosis task
**Deliverable**: Evaluated on MedQA-USMLE (200 cases) + DDXPlus (100 cases).

### Promise 5: Structured report generation task
**Deliverable**: ReportWriter agent produces structured clinical reports;
rubric-based evaluation scores each report on 8 dimensions.

### Promise 6: Ablation study isolating marginal contribution of each layer
**Deliverable**: 4 configurations tested independently:
- BASE (no governance)
- BASE + Verifier only
- BASE + HITL only
- Combined (full stack)

This isolates each component's ΔCSS and ΔCost separately.

### Promise 7: Marginal cost-benefit curves for each oversight mechanism
**Deliverable**: GE = ΔCSS/ΔCCS computed per component and per cumulative level.
Pareto frontier plotted as CSS vs CCS.

### Promise 8: Open-source measurement harness
**Deliverable**: Full GitHub repo + Colab notebook. Reproducible on free T4 GPU.
No proprietary APIs required.

### Promise 9: Regulatory framework discussion
**Deliverable**: Section discussing risk-proportional governance, quantified
compliance evidence, cost of over-governance, and 5 recommendations for regulators.

---

## Three Novel Contributions (what reviewers will evaluate)

### Contribution 1: Governance-as-a-dial
We operationalize AI governance as a 5-level taxonomy (G0–G4) that is:
- Mechanically distinct at each level (not just "more prompting")
- Applicable to any multi-agent LLM framework
- Reproducible across models and datasets

**Why this is new**: Prior work treats governance as binary (guardrails on/off).
We treat it as a continuous spectrum with measurable granularity.

### Contribution 2: Safety is not accuracy
We introduce a *Clinical Safety Score (CSS)* that separates:
- Critical Miss Rate (CMR): missing a life-threatening diagnosis
- Hallucination Impact Rate (HIR): fabricated findings that change the output
- Unsafe Reassurance Rate (URR): overconfident wrong diagnoses

**Why this is new**: All cost-tradeoff papers use accuracy as the quality axis.
We use harm-weighted safety — a much more clinically meaningful signal.

### Contribution 3: Governance Efficiency (GE) as a deployment metric
We define:
    GE = ΔCSS / ΔCCS

GE tells a hospital exactly: "for every unit of extra compute you spend on
governance, you gain X% in clinical safety."

**Why this is new**: No existing paper provides a single metric for governance ROI.

---

## How We Differ From the Three Closest Papers

| Paper | What they do | Our difference |
|-------|-------------|----------------|
| TeamMedAgents (2025) | Pareto frontier of accuracy vs. token cost | We use *safety* (not accuracy) as Y-axis, *governance level* as X variable |
| ConfAgents (2025) | Adaptive cost-accuracy tradeoff | They adapt based on case difficulty; we vary governance mechanisms systematically |
| MedSafe-Dx (2026) | Safety benchmark for single-agent LLMs | Single-agent only; no multi-agent governance, no cost measurement |
| ETHOS (2026) | Ethics framework for clinical MAS | Conceptual framework; we provide empirical cost measurements |
| npj Digital Medicine (2026) | Benchmarking agent systems for clinical tasks | Measures agent overhead; we isolate governance layers specifically |

---

## IEEE Access Framing
- **Scope**: Biomedical informatics + AI systems evaluation
- **Contribution type**: Benchmark + empirical study + open-source harness
- **Reproducibility**: Full open-source, no proprietary APIs required
- **Impact statement**: Provides deployment guidance for healthcare AI governance
- **Regulatory relevance**: Connects empirical findings to FDA/EU AI Act/NIST frameworks
