"""
Governance levels G0–G4.
Each returns a GovernanceResult with full cost and decision trace.
"""
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from src.agents.base import BaseAgent, AgentResponse
from src.agents.prompts import (
    DIAGNOSTICIAN_PROMPT, SPECIALIST_PROMPT,
    VERIFIER_PROMPT, MODERATOR_PROMPT, ETHICS_CRITIC_PROMPT,
    PROMPT_VERSION,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GovernanceResult:
    case_id: str
    governance_level: str        # "G0"–"G4"
    model: str
    seed: int
    prompt_version: str

    # Final output
    top_diagnosis: Optional[str] = None
    top3_diagnoses: list = field(default_factory=list)
    output_confidence: Optional[float] = None
    abstained: bool = False
    suppressed: bool = False

    # Safety signals (filled by evaluator, not the governance level)
    hallucination_detected: bool = False
    hallucination_impactful: bool = False

    # Cost
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_latency: float = 0.0
    agent_turns: int = 0

    # Trace (every agent call, in order)
    trace: list = field(default_factory=list)

    # Ethics critic result (G4 only)
    ethics_verdict: Optional[str] = None
    ethics_score: Optional[int] = None

    def add_turn(self, resp: AgentResponse):
        self.total_input_tokens += resp.input_tokens
        self.total_output_tokens += resp.output_tokens
        self.total_tokens += resp.input_tokens + resp.output_tokens
        self.total_latency += resp.latency
        self.agent_turns += 1
        self.trace.append({
            "agent": resp.agent_role,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "latency": resp.latency,
            "text": resp.text[:500],   # truncated for log size
        })

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> Optional[dict]:
    """Extract JSON from LLM response, even if surrounded by prose."""
    try:
        # Try direct parse first
        return json.loads(text.strip())
    except Exception:
        pass
    # Fallback: find first { ... } block
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return None


def _build_case_prompt(case: dict) -> str:
    """Convert a case dict into a structured prompt string."""
    lines = [f"Patient Case ID: {case.get('id', 'unknown')}"]
    if "age" in case:
        lines.append(f"Age: {case['age']}")
    if "sex" in case:
        lines.append(f"Sex: {case['sex']}")
    if "chief_complaint" in case:
        lines.append(f"Chief Complaint: {case['chief_complaint']}")
    if "history" in case:
        lines.append(f"History: {case['history']}")
    if "symptoms" in case:
        symptoms = case["symptoms"]
        if isinstance(symptoms, list):
            symptoms = ", ".join(symptoms)
        lines.append(f"Symptoms: {symptoms}")
    if "vitals" in case:
        lines.append(f"Vitals: {case['vitals']}")
    if "labs" in case:
        lines.append(f"Labs: {case['labs']}")
    if "question" in case:
        lines.append(f"\nQuestion: {case['question']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# G0 — Bare Multi-Agent (no governance)
# ---------------------------------------------------------------------------

def run_g0(case: dict, model: str, seed: int) -> GovernanceResult:
    result = GovernanceResult(
        case_id=case["id"], governance_level="G0",
        model=model, seed=seed, prompt_version=PROMPT_VERSION,
    )
    case_text = _build_case_prompt(case)

    agent_a = BaseAgent(model, "diagnostician_a", DIAGNOSTICIAN_PROMPT)
    agent_b = BaseAgent(model, "diagnostician_b", DIAGNOSTICIAN_PROMPT)

    resp_a = agent_a.call(case_text)
    result.add_turn(resp_a)
    resp_b = agent_b.call(case_text)
    result.add_turn(resp_b)

    parsed_a = _parse_json(resp_a.text) or {}
    parsed_b = _parse_json(resp_b.text) or {}

    diag_a = parsed_a.get("top_diagnosis")
    diag_b = parsed_b.get("top_diagnosis")
    conf_a = parsed_a.get("confidence", 0.5)
    conf_b = parsed_b.get("confidence", 0.5)

    # Majority: if both agree, use either; else use A (tie-break)
    if diag_a and diag_b and diag_a.lower() == diag_b.lower():
        result.top_diagnosis = diag_a
        result.output_confidence = (conf_a + conf_b) / 2
    else:
        result.top_diagnosis = diag_a  # fallback to A
        result.output_confidence = conf_a

    # Build top3 from both differentials
    seen = set()
    for parsed in [parsed_a, parsed_b]:
        for item in parsed.get("differential", []):
            d = item.get("diagnosis")
            if d and d not in seen:
                result.top3_diagnoses.append(d)
                seen.add(d)
            if len(result.top3_diagnoses) >= 3:
                break

    return result


# ---------------------------------------------------------------------------
# G1 — Structured Roles + Basic Guardrails
# ---------------------------------------------------------------------------

def run_g1(case: dict, model: str, seed: int) -> GovernanceResult:
    result = GovernanceResult(
        case_id=case["id"], governance_level="G1",
        model=model, seed=seed, prompt_version=PROMPT_VERSION,
    )
    case_text = _build_case_prompt(case)

    generalist = BaseAgent(model, "generalist", DIAGNOSTICIAN_PROMPT)
    specialist = BaseAgent(model, "specialist", SPECIALIST_PROMPT)

    resp_g = generalist.call(case_text)
    result.add_turn(resp_g)
    resp_s = specialist.call(case_text)
    result.add_turn(resp_s)

    parsed_s = _parse_json(resp_s.text) or {}
    parsed_g = _parse_json(resp_g.text) or {}

    # Rule-based guardrail: specialist output must have a non-empty primary_diagnosis
    specialist_diag = parsed_s.get("primary_diagnosis", "").strip()
    specialist_conf = parsed_s.get("confidence", 0.0)

    if specialist_diag and len(specialist_diag) > 2:
        result.top_diagnosis = specialist_diag
        result.output_confidence = specialist_conf
    else:
        # Fallback to generalist
        result.top_diagnosis = parsed_g.get("top_diagnosis")
        result.output_confidence = parsed_g.get("confidence", 0.5)

    seen = set()
    for d in [specialist_diag] + [i.get("diagnosis", "") for i in parsed_g.get("differential", [])]:
        if d and d not in seen:
            result.top3_diagnoses.append(d)
            seen.add(d)
        if len(result.top3_diagnoses) >= 3:
            break

    return result


# ---------------------------------------------------------------------------
# G2 — + Verifier Agent (hallucination check)
# ---------------------------------------------------------------------------

def run_g2(case: dict, model: str, seed: int) -> GovernanceResult:
    result = GovernanceResult(
        case_id=case["id"], governance_level="G2",
        model=model, seed=seed, prompt_version=PROMPT_VERSION,
    )
    case_text = _build_case_prompt(case)

    generalist = BaseAgent(model, "generalist", DIAGNOSTICIAN_PROMPT)
    specialist = BaseAgent(model, "specialist", SPECIALIST_PROMPT)
    verifier = BaseAgent(model, "verifier", VERIFIER_PROMPT)

    resp_g = generalist.call(case_text)
    result.add_turn(resp_g)
    resp_s = specialist.call(case_text)
    result.add_turn(resp_s)

    parsed_s = _parse_json(resp_s.text) or {}
    specialist_diag = parsed_s.get("primary_diagnosis", "").strip()
    reasoning = parsed_s.get("reasoning", "")

    # Verify the specialist's reasoning against the case facts
    verify_input = (
        f"PATIENT CASE:\n{case_text}\n\n"
        f"PROPOSED DIAGNOSIS: {specialist_diag}\n"
        f"REASONING: {reasoning}"
    )
    resp_v = verifier.call(verify_input)
    result.add_turn(resp_v)

    parsed_v = _parse_json(resp_v.text) or {}
    verdict = parsed_v.get("verdict", "VERIFIED")
    fabricated_count = parsed_v.get("fabricated_count", 0)

    if verdict == "FLAGGED" and fabricated_count > 0:
        # Strip hallucinated claims, re-run with just case facts
        result.hallucination_detected = True
        # If fabricated findings > 1, treat as impactful
        result.hallucination_impactful = fabricated_count > 1

        # Fallback to generalist output
        parsed_g = _parse_json(resp_g.text) or {}
        result.top_diagnosis = parsed_g.get("top_diagnosis")
        result.output_confidence = parsed_g.get("confidence", 0.4)
    else:
        result.top_diagnosis = specialist_diag if specialist_diag else (
            _parse_json(resp_g.text) or {}).get("top_diagnosis")
        result.output_confidence = parsed_s.get("confidence", 0.5)

    # Build top3
    parsed_g = _parse_json(resp_g.text) or {}
    seen = set()
    for d in [result.top_diagnosis] + [
        i.get("diagnosis", "") for i in parsed_g.get("differential", [])
    ]:
        if d and d not in seen:
            result.top3_diagnoses.append(d)
            seen.add(d)
        if len(result.top3_diagnoses) >= 3:
            break

    return result


# ---------------------------------------------------------------------------
# G3 — + Consensus + Confidence Threshold
# ---------------------------------------------------------------------------

def run_g3(case: dict, model: str, seed: int) -> GovernanceResult:
    result = GovernanceResult(
        case_id=case["id"], governance_level="G3",
        model=model, seed=seed, prompt_version=PROMPT_VERSION,
    )
    case_text = _build_case_prompt(case)

    generalist = BaseAgent(model, "generalist", DIAGNOSTICIAN_PROMPT)
    specialist_a = BaseAgent(model, "specialist_a", SPECIALIST_PROMPT)
    specialist_b = BaseAgent(model, "specialist_b", SPECIALIST_PROMPT)
    moderator = BaseAgent(model, "moderator", MODERATOR_PROMPT)

    resp_g = generalist.call(case_text)
    result.add_turn(resp_g)
    resp_sa = specialist_a.call(case_text)
    result.add_turn(resp_sa)
    resp_sb = specialist_b.call(case_text)
    result.add_turn(resp_sb)

    parsed_g = _parse_json(resp_g.text) or {}
    parsed_sa = _parse_json(resp_sa.text) or {}
    parsed_sb = _parse_json(resp_sb.text) or {}

    agent_votes = [
        {"agent": "generalist", "diagnosis": parsed_g.get("top_diagnosis", ""), "confidence": parsed_g.get("confidence", 0.5)},
        {"agent": "specialist_a", "diagnosis": parsed_sa.get("primary_diagnosis", ""), "confidence": parsed_sa.get("confidence", 0.5)},
        {"agent": "specialist_b", "diagnosis": parsed_sb.get("primary_diagnosis", ""), "confidence": parsed_sb.get("confidence", 0.5)},
    ]

    moderator_input = f"Agent votes:\n{json.dumps(agent_votes, indent=2)}"
    resp_m = moderator.call(moderator_input)
    result.add_turn(resp_m)

    parsed_m = _parse_json(resp_m.text) or {}
    recommendation = parsed_m.get("recommendation", "ABSTAIN")

    if recommendation == "REPORT":
        result.top_diagnosis = parsed_m.get("agreed_diagnosis")
        result.output_confidence = parsed_m.get("average_confidence", 0.7)
    elif recommendation == "SECOND_ROUND":
        # One more round — run moderator again with explicit tie-breaking
        resp_m2 = moderator.call(
            moderator_input + "\n\nThis is round 2. If no consensus, recommend ABSTAIN."
        )
        result.add_turn(resp_m2)
        parsed_m2 = _parse_json(resp_m2.text) or {}
        if parsed_m2.get("recommendation") == "REPORT":
            result.top_diagnosis = parsed_m2.get("agreed_diagnosis")
            result.output_confidence = parsed_m2.get("average_confidence", 0.6)
        else:
            result.abstained = True
    else:  # ABSTAIN
        result.abstained = True

    # Build top3 from all agent outputs
    seen = set()
    for d in [result.top_diagnosis] + [v["diagnosis"] for v in agent_votes]:
        if d and d not in seen:
            result.top3_diagnoses.append(d)
            seen.add(d)
        if len(result.top3_diagnoses) >= 3:
            break

    return result


# ---------------------------------------------------------------------------
# G4 — + Ethics Critic + Audit Logging
# ---------------------------------------------------------------------------

def run_g4(case: dict, model: str, seed: int) -> GovernanceResult:
    # G4 builds on G3 — run G3 first
    result = run_g3(case, model, seed)
    result.governance_level = "G4"

    if result.abstained:
        # Already escalated — no need for ethics critic
        return result

    ethics_critic = BaseAgent(model, "ethics_critic", ETHICS_CRITIC_PROMPT)

    critic_input = (
        f"PATIENT CASE:\n{_build_case_prompt(case)}\n\n"
        f"PROPOSED OUTPUT:\n"
        f"Top Diagnosis: {result.top_diagnosis}\n"
        f"Confidence: {result.output_confidence}\n"
        f"Top 3: {', '.join(result.top3_diagnoses)}"
    )

    resp_e = ethics_critic.call(critic_input)
    result.add_turn(resp_e)

    parsed_e = _parse_json(resp_e.text) or {}
    verdict = parsed_e.get("verdict", "APPROVED")
    total_score = parsed_e.get("total_score", 20)

    result.ethics_verdict = verdict
    result.ethics_score = total_score

    if verdict == "REQUIRES_REVISION":
        # One revision attempt — re-run generalist with ethics guidance
        revision_note = parsed_e.get("required_revision", "")
        generalist = BaseAgent(model, "generalist_revised", DIAGNOSTICIAN_PROMPT)
        revised_input = (
            f"{_build_case_prompt(case)}\n\n"
            f"IMPORTANT REVISION NOTE: {revision_note}"
        )
        resp_rev = generalist.call(revised_input)
        result.add_turn(resp_rev)
        parsed_rev = _parse_json(resp_rev.text) or {}
        result.top_diagnosis = parsed_rev.get("top_diagnosis", result.top_diagnosis)
        result.output_confidence = parsed_rev.get("confidence", result.output_confidence) * 0.9

        # Re-evaluate
        resp_e2 = ethics_critic.call(
            f"PATIENT CASE:\n{_build_case_prompt(case)}\n\n"
            f"REVISED OUTPUT:\nTop Diagnosis: {result.top_diagnosis}\n"
            f"Confidence: {result.output_confidence}"
        )
        result.add_turn(resp_e2)
        parsed_e2 = _parse_json(resp_e2.text) or {}
        if parsed_e2.get("verdict") == "SUPPRESSED":
            result.suppressed = True
            result.top_diagnosis = None

    elif verdict == "SUPPRESSED":
        result.suppressed = True
        result.top_diagnosis = None

    return result


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

GOVERNANCE_RUNNERS = {
    "G0": run_g0,
    "G1": run_g1,
    "G2": run_g2,
    "G3": run_g3,
    "G4": run_g4,
}

def run_governance(level: str, case: dict, model: str, seed: int) -> GovernanceResult:
    if level not in GOVERNANCE_RUNNERS:
        raise ValueError(f"Unknown governance level: {level}")
    return GOVERNANCE_RUNNERS[level](case, model, seed)
