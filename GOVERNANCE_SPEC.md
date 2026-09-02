# Governance Level Specification: G0 → G4

This is the core independent variable of the study.
Each level adds one concrete mechanism. They are *cumulative* — G3 includes everything in G1+G2.

---

## G0 — Bare Multi-Agent (Baseline)
**What it is**: Two agents, no oversight. Just parallel opinions that get averaged.

**Agents**:
- Agent A (Diagnostician 1): Reads case, outputs ranked differential diagnosis
- Agent B (Diagnostician 2): Same task, independent

**Decision rule**: Majority vote. If they agree → output. If they disagree → pick Agent A.

**Governance mechanisms**: None

**Expected cost**: ~2× single agent (2 LLM calls per case)

**Implementation**:
```python
# Pseudocode
resp_a = llm.call(system=DIAGNOSTICIAN_PROMPT, user=case)
resp_b = llm.call(system=DIAGNOSTICIAN_PROMPT, user=case)
output = majority_vote([resp_a, resp_b])
```

---

## G1 — Structured Roles + Basic Guardrails
**What it is**: Agents are given distinct roles. A simple rule-based filter blocks
obviously bad outputs (empty diagnosis, diagnosis not in known ICD-10 list).

**Agents**:
- Generalist Agent: Broad differential, 3–5 diagnoses
- Specialist Agent: Focuses on the most likely top-1 diagnosis with reasoning
- Rule-based Guardrail (NOT an LLM — just code): Validates output format, rejects
  responses that contain no ICD-10 codes or flag self-harm/harmful advice

**Decision rule**: Specialist's top-1 if valid; fallback to Generalist's top-1.

**Governance mechanisms**:
1. Role specialization (prompt-level)
2. Deterministic output validator (code-level)

**Expected cost**: ~2.2× single agent

---

## G2 — Verifier Agent (Hallucination Check)
**What it is**: A third agent whose only job is to fact-check the diagnosis
against the provided case facts. It flags hallucinated symptoms or fabricated findings.

**Agents** (from G1, plus):
- Verifier Agent: Receives (case_facts, proposed_diagnosis, reasoning_chain) and
  outputs a VERIFIED / FLAGGED decision with a list of unsupported claims

**Decision rule**: If FLAGGED → the reasoning chain is stripped of unsupported claims
and the diagnosis is re-evaluated by the Generalist. If VERIFIED → pass through.

**Governance mechanisms** (from G1, plus):
3. LLM-as-verifier for hallucination detection

**Expected cost**: ~3× single agent (3 LLM calls in main path, up to 4 if flagged)

**Key metric this level targets**: Hallucination Impact Rate (HIR)

---

## G3 — Consensus + Confidence Threshold
**What it is**: Adds a formal consensus round where agents must reach agreement
above a confidence threshold. Cases that can't reach consensus are flagged for
human escalation (simulated in benchmark as "abstain").

**Agents** (from G2, plus):
- Consensus Moderator: Receives all agent outputs, asks each agent to state a
  confidence score (0–1) and their top diagnosis. Checks: do ≥2 agents agree on
  top-1 with confidence ≥ 0.7? If yes → output with high confidence flag.
  If no → second deliberation round or abstain.

**Decision rule**:
- Consensus reached (≥2 agree, conf ≥ 0.7) → output diagnosis
- No consensus after 2 rounds → ABSTAIN (escalate to human)

**Governance mechanisms** (from G2, plus):
4. Confidence-gated consensus
5. Abstention / escalation mechanism

**Expected cost**: ~4–5× single agent (more in contested cases)

**Key metric this level targets**: Unsafe Reassurance Rate (URR) — the system
now refuses to give a confident answer when it's uncertain, which directly
reduces overconfident wrong diagnoses.

---

## G4 — Full Governance Stack (Ethics + Audit)
**What it is**: Adds an Ethics Critic agent that evaluates the final output
against a safety rubric (similar to ETHOS framework), plus full audit logging
of every agent turn.

**Agents** (from G3, plus):
- Ethics Critic: Receives the consensus output and evaluates:
  1. Is this diagnosis safe to communicate to a clinician? (beneficence check)
  2. Are appropriate uncertainties communicated?
  3. Is there any advice that could cause direct harm if acted on?
  If it fails → the output is revised once. If still fails → SUPPRESS + log.
- Audit Logger (not an LLM): Records every agent turn, confidence score, and
  decision point to a structured JSON log.

**Decision rule**: Output only if Ethics Critic approves. Otherwise suppress
and return a "unable to safely diagnose — escalate to human" response.

**Governance mechanisms** (from G3, plus):
6. LLM-as-ethics-critic with rubric
7. Structured audit trail
8. Suppression mechanism for unsafe outputs

**Expected cost**: ~5–7× single agent

**Key metric this level targets**: Critical Miss Rate (CMR) — high-acuity cases
that would have been confidently wrong at G0 are now either correctly handled or
safely escalated.

---

## Summary Table

| Level | Agents | Key Mechanism Added | Primary Safety Target | Approx Cost Multiplier |
|-------|--------|--------------------|-----------------------|------------------------|
| G0 | 2 | None (bare baseline) | — | 2× |
| G1 | 2 + rules | Role specialization + output validation | Format errors | 2.2× |
| G2 | 3 | Hallucination verifier | HIR | 3× |
| G3 | 4 | Consensus + confidence + abstention | URR | 4–5× |
| G4 | 5 + audit | Ethics critic + suppression + logging | CMR | 5–7× |

---

## Implementation Notes

### Prompt templates needed
- `DIAGNOSTICIAN_PROMPT` — system prompt for diagnostic agents
- `SPECIALIST_PROMPT` — system prompt biased toward single-disease focus
- `VERIFIER_PROMPT` — system prompt for hallucination detection
- `MODERATOR_PROMPT` — system prompt for consensus moderation
- `ETHICS_CRITIC_PROMPT` — system prompt with safety rubric

### What stays constant across levels
- The underlying LLM (same model, same temperature=0.0 for reproducibility)
- The case format (same structured JSON input)
- The evaluation metrics (measured identically for all levels)
- Random seeds (3 seeds: 42, 123, 777)

### What varies across levels
- Number of LLM calls
- Agent prompts / roles
- Decision-making logic
- Whether abstention is possible
