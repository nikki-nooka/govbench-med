"""
Phase 2: Experiment Telemetry Module
Complete telemetry schema for GovBench-Med experiments.
"""
import os
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import hashlib


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    FLAGGED = "FLAGGED"
    NOT_CHECKED = "NOT_CHECKED"


class GovernanceLevel(str, Enum):
    G0 = "G0"
    G1 = "G1"
    G2 = "G2"
    G3 = "G3"
    G4 = "G4"


class ModelFamily(str, Enum):
    LLAMA = "llama"
    MISTRAL = "mistral"
    QWEN = "qwen"


# ---------------------------------------------------------------------------
# Telemetry Data Classes
# ---------------------------------------------------------------------------

@dataclass
class AgentTurn:
    """Single agent turn with full cost and content trace."""
    agent_role: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    text: str
    model: str
    temperature: float
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerificationTrace:
    status: str
    claims_checked: int
    supported_claims: int
    contradicted_claims: int
    unverifiable_claims: int
    fabricated_count: int
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConsensusTrace:
    consensus_reached: bool
    agreed_diagnosis: str
    average_confidence: float
    recommendation: str
    agent_votes: List[Dict[str, Any]]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EthicsTrace:
    verdict: str
    scores: Dict[str, int]
    total_score: int
    issues: List[str]
    required_revision: str
    revision_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReportTrace:
    clinical_report: Optional[Dict] = None
    report_evaluation: Optional[Dict] = None
    report_gen_latency_ms: float = 0.0
    report_eval_latency_ms: float = 0.0
    report_total_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunTelemetry:
    # Identity
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    experiment_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    prompt_version: str = "v1.1"
    code_version: str = "0.1.0"

    # Experiment context
    case_id: str = ""
    dataset: str = ""
    model: str = ""
    model_version: str = ""
    governance_level: str = "G0"
    seed: int = 42

    # Outputs
    final_answer: str = ""
    ground_truth: str = ""
    ground_truth_key: str = ""
    dataset_source: str = ""

    # Quality
    correctness: bool = False
    top3_correct: bool = False
    confidence: float = 0.0
    consensus: bool = False
    abstained: bool = False
    suppressed: bool = False

    # Governance mechanics
    verification_status: str = "NOT_CHECKED"
    claims_checked: int = 0
    supported_claims: int = 0
    contradicted_claims: int = 0
    unverifiable_claims: int = 0
    revision_count: int = 0

    # Cost & Latency
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model_calls: int = 0
    latency_ms: float = 0.0
    governance_latency_ms: float = 0.0
    diagnosis_latency_ms: float = 0.0
    governance_overhead_latency_ms: float = 0.0
    report_latency_ms: float = 0.0

    # Cost
    cost: float = 0.0

    # Human review
    human_review_required: bool = False
    human_review_time: float = 0.0

    # Governance metrics
    css: float = 0.0
    cmr: float = 0.0
    hir: float = 0.0
    urr: float = 0.0
    ccs: float = 0.0
    ge: float = 0.0

    # Safety flags
    hallucination_impactful: bool = False
    hallucination_detected: bool = False
    unsafe_reassurance: bool = False
    abstained: bool = False
    suppressed: bool = False

    # Traces
    agent_traces: List[Dict] = field(default_factory=list)
    verification_trace: Optional[Dict] = None
    consensus_trace: Optional[Dict] = None
    ethics_trace: Optional[Dict] = None
    report_trace: Optional[Dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_csv_row(self) -> dict:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "case_id": self.case_id,
            "dataset": self.dataset,
            "model": self.model,
            "model_version": self.model_version,
            "governance_level": self.governance_level,
            "seed": self.seed,
            "final_answer": self.final_answer,
            "ground_truth": self.ground_truth,
            "ground_truth_key": self.ground_truth_key,
            "dataset_source": self.dataset_source,
            "correctness": self.correctness,
            "top3_correct": self.top3_correct,
            "critical_miss": self.critical_miss if hasattr(self, "critical_miss") else False,
            "confidence": self.confidence,
            "consensus": self.consensus,
            "abstained": self.abstained,
            "suppressed": self.suppressed,
            "verification_status": self.verification_status,
            "claims_checked": self.claims_checked,
            "supported_claims": self.supported_claims,
            "contradicted_claims": self.contradicted_claims,
            "unverifiable_claims": self.unverifiable_claims,
            "revision_count": self.revision_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "model_calls": self.model_calls,
            "latency_ms": self.latency_ms,
            "governance_latency_ms": self.governance_latency_ms,
            "diagnosis_latency_ms": self.diagnosis_latency_ms,
            "governance_overhead_latency_ms": self.governance_overhead_latency_ms,
            "report_latency_ms": self.report_latency_ms,
            "cost": self.cost,
            "human_review_required": self.human_review_required,
            "human_review_time": self.human_review_time,
            "css": self.css,
            "cmr": self.cmr,
            "hir": self.hir,
            "urr": self.urr,
            "ccs": self.ccs,
            "ge": self.ge,
            "hallucination_impactful": self.hallucination_impactful,
            "hallucination_detected": self.hallucination_detected,
            "unsafe_reassurance": self.unsafe_reassurance,
            "abstained": self.abstained,
            "suppressed": self.suppressed,
        }

    @classmethod
    def from_governance_result(cls, result, case: dict, model: str, level: str, seed: int, evaluator) -> "RunTelemetry":
        # Parse traces
        verification = None
        consensus = None
        ethics = None

        for trace in result.trace:
            if trace.get("agent") == "verifier":
                try:
                    parsed = json.loads(trace["text"])
                    verification = parsed
                except:
                    pass
            elif trace.get("agent") == "moderator":
                try:
                    parsed = json.loads(trace["text"])
                    consensus = parsed
                except:
                    pass
            elif trace.get("agent") == "ethics_critic":
                try:
                    parsed = json.loads(trace["text"])
                    ethics = parsed
                except:
                    pass

        hallucination_impactful = getattr(result, "hallucination_impactful", False)
        hallucination_detected = getattr(result, "hallucination_detected", False)

        telemetry = cls(
            run_id=str(uuid.uuid4())[:8],
            experiment_id=f"{case['id']}_{model}_{level}_seed{seed}",
            case_id=case["id"],
            dataset=case.get("source", "unknown"),
            model=model,
            model_version=model,
            governance_level=level,
            seed=seed,
            final_answer=result.top_diagnosis or "",
            ground_truth=case.get("ground_truth", ""),
            ground_truth_key=case.get("ground_truth_key", ""),
            dataset_source=case.get("source", "unknown"),
            input_tokens=result.total_input_tokens,
            output_tokens=result.total_output_tokens,
            total_tokens=result.total_tokens,
            model_calls=result.agent_turns,
            latency_ms=result.total_latency * 1000.0,
            governance_latency_ms=result.total_latency * 1000.0,
            diagnosis_latency_ms=result.total_latency * 600.0,
            governance_overhead_latency_ms=result.total_latency * 400.0,
            report_latency_ms=getattr(result, "report_gen_latency", 0) + getattr(result, "report_eval_latency", 0),
            human_review_required=result.abstained or result.suppressed,
            verification_status=verification.get("verdict", "NOT_CHECKED") if verification else "NOT_CHECKED",
            claims_checked=verification.get("claims", 0) if verification else 0,
            supported_claims=sum(1 for c in verification.get("claims", []) if c.get("status") == "SUPPORTED") if verification else 0,
            contradicted_claims=sum(1 for c in verification.get("claims", []) if c.get("status") == "CONTRADICTED") if verification else 0,
            unverifiable_claims=sum(1 for c in verification.get("claims", []) if c.get("status") not in ["SUPPORTED", "CONTRADICTED"]) if verification else 0,
            revision_count=ethics.get("revision_count", 0) if ethics else 0,
            hallucination_impactful=getattr(result, "hallucination_impactful", False),
            hallucination_detected=getattr(result, "hallucination_detected", False),
            abstained=result.abstained,
            suppressed=result.suppressed,
            agent_traces=result.trace,
            verification_trace=verification,
            consensus_trace=consensus,
            ethics_trace=ethics,
        )

        # Score
        try:
            scored = evaluator.score(result, case.get("ground_truth", ""))
            telemetry.correctness = scored.top1_correct
            telemetry.top3_correct = scored.top3_correct
            telemetry.confidence = result.output_confidence or 0.0
            telemetry.consensus = not result.abstained
            telemetry.abstained = result.abstained
            telemetry.suppressed = result.suppressed
            telemetry.human_review_required = result.abstained or result.suppressed
            telemetry.hallucination_impactful = scored.hallucination_impactful
            telemetry.unsafe_reassurance = scored.unsafe_reassurance
        except Exception:
            pass

        return telemetry


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import hashlib
import re
from datetime import datetime


# Enums
class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    FLAGGED = "FLAGGED"
    NOT_CHECKED = "NOT_CHECKED"

class GovernanceLevel(str, Enum):
    G0 = "G0"
    G1 = "G1"
    G2 = "G2"
    G3 = "G3"
    G4 = "G4"

class ModelFamily(str, Enum):
    LLAMA = "llama"
    MISTRAL = "mistral"
    QWEN = "qwen"

# AgentTurn
@dataclass
class AgentTurn:
    agent_role: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    text: str
    model: str
    temperature: float
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)

# VerificationTrace
@dataclass
class VerificationTrace:
    status: str
    claims_checked: int
    supported_claims: int
    contradicted_claims: int
    unverifiable_claims: int
    fabricated_count: int
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)

# ConsensusTrace
@dataclass
class ConsensusTrace:
    consensus_reached: bool
    agreed_diagnosis: str
    average_confidence: float
    recommendation: str
    agent_votes: List[Dict[str, Any]]

    def to_dict(self) -> dict:
        return asdict(self)

# EthicsTrace
@dataclass
class EthicsTrace:
    verdict: str
    scores: Dict[str, int]
    total_score: int
    issues: List[str]
    required_revision: str
    revision_count: int

    def to_dict(self) -> dict:
        return asdict(self)

# ReportTrace
@dataclass
class ReportTrace:
    clinical_report: Optional[Dict] = None
    report_evaluation: Optional[Dict] = None
    report_gen_latency_ms: float = 0.0
    report_eval_latency_ms: float = 0.0
    report_total_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

# RunTelemetry
@dataclass
class RunTelemetry:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    experiment_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    prompt_version: str = "v1.1"
    code_version: str = "0.1.0"

    case_id: str = ""
    dataset: str = ""
    model: str = ""
    model_version: str = ""
    governance_level: str = "G0"
    seed: int = 42

    final_answer: str = ""
    ground_truth: str = ""
    ground_truth_key: str = ""
    dataset_source: str = ""

    correctness: bool = False
    top3_correct: bool = False
    confidence: float = 0.0
    consensus: bool = False
    abstained: bool = False
    suppressed: bool = False

    verification_status: str = "NOT_CHECKED"
    claims_checked: int = 0
    supported_claims: int = 0
    contradicted_claims: int = 0
    unverifiable_claims: int = 0
    revision_count: int = 0

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model_calls: int = 0
    latency_ms: float = 0.0
    governance_latency_ms: float = 0.0
    diagnosis_latency_ms: float = 0.0
    governance_overhead_latency_ms: float = 0.0
    report_latency_ms: float = 0.0

    cost: float = 0.0
    human_review_required: bool = False
    human_review_time: float = 0.0

    css: float = 0.0
    cmr: float = 0.0
    hir: float = 0.0
    urr: float = 0.0
    ccs: float = 0.0
    ge: float = 0.0

    hallucination_impactful: bool = False
    hallucination_detected: bool = False
    unsafe_reassurance: bool = False
    abstained: bool = False
    suppressed: bool = False

    agent_traces: List[Dict] = field(default_factory=list)
    verification_trace: Optional[Dict] = None
    consensus_trace: Optional[Dict] = None
    ethics_trace: Optional[Dict] = None
    report_trace: Optional[Dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_csv_row(self) -> dict:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "case_id": self.case_id,
            "dataset": self.dataset,
            "model": self.model,
            "model_version": self.model_version,
            "governance_level": self.governance_level,
            "seed": self.seed,
            "final_answer": self.final_answer,
            "ground_truth": self.ground_truth,
            "ground_truth_key": self.ground_truth_key,
            "dataset_source": self.dataset_source,
            "correctness": self.correctness,
            "top3_correct": self.top3_correct,
            "critical_miss": self.critical_miss if hasattr(self, "critical_miss") else False,
            "confidence": self.confidence,
            "consensus": self.consensus,
            "abstained": self.abstained,
            "suppressed": self.suppressed,
            "verification_status": self.verification_status,
            "claims_checked": self.claims_checked,
            "supported_claims": self.supported_claims,
            "contradicted_claims": self.contradicted_claims,
            "unverifiable_claims": self.unverifiable_claims,
            "revision_count": self.revision_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "model_calls": self.model_calls,
            "latency_ms": self.latency_ms,
            "governance_latency_ms": self.governance_latency_ms,
            "diagnosis_latency_ms": self.diagnosis_latency_ms,
            "governance_overhead_latency_ms": self.governance_overhead_latency_ms,
            "report_latency_ms": self.report_latency_ms,
            "cost": self.cost,
            "human_review_required": self.human_review_required,
            "human_review_time": self.human_review_time,
            "css": self.css,
            "cmr": self.cmr,
            "hir": self.hir,
            "urr": self.urr,
            "ccs": self.ccs,
            "ge": self.ge,
            "hallucination_impactful": self.hallucination_impactful,
            "hallucination_detected": self.hallucination_detected,
            "unsafe_reassurance": self.unsafe_reassurance,
            "abstained": self.abstained,
            "suppressed": self.suppressed,
        }

    @classmethod
    def from_governance_result(cls, result, case: dict, model: str, level: str, seed: int, evaluator) -> "RunTelemetry":
        verification = None
        consensus = None
        ethics = None

        for trace in result.trace:
            if trace.get("agent") == "verifier":
                try:
                    parsed = json.loads(trace["text"])
                    verification = parsed
                except:
                    pass
            elif trace.get("agent") == "moderator":
                try:
                    parsed = json.loads(trace["text"])
                    consensus = parsed
                except:
                    pass
            elif trace.get("agent") == "ethics_critic":
                try:
                    parsed = json.loads(trace["text"])
                    ethics = parsed
                except:
                    pass

        hallucination_impactful = getattr(result, "hallucination_impactful", False)
        hallucination_detected = getattr(result, "hallucination_detected", False)

        telemetry = cls(
            run_id=str(uuid.uuid4())[:8],
            experiment_id=f"{case['id']}_{model}_{level}_seed{seed}",
            case_id=case["id"],
            dataset=case.get("source", "unknown"),
            model=model,
            model_version=model,
            governance_level=level,
            seed=seed,
            final_answer=result.top_diagnosis or "",
            ground_truth=case.get("ground_truth", ""),
            ground_truth_key=case.get("ground_truth_key", ""),
            dataset_source=case.get("source", "unknown"),
            input_tokens=result.total_input_tokens,
            output_tokens=result.total_output_tokens,
            total_tokens=result.total_tokens,
            model_calls=result.agent_turns,
            latency_ms=result.total_latency * 1000.0,
            governance_latency_ms=result.total_latency * 1000.0,
            diagnosis_latency_ms=result.total_latency * 600.0,
            governance_overhead_latency_ms=result.total_latency * 400.0,
            report_latency_ms=getattr(result, "report_gen_latency", 0) + getattr(result, "report_eval_latency", 0),
            human_review_required=result.abstained or result.suppressed,
            verification_status=verification.get("verdict", "NOT_CHECKED") if verification else "NOT_CHECKED",
            claims_checked=verification.get("claims", 0) if verification else 0,
            supported_claims=sum(1 for c in verification.get("claims", []) if c.get("status") == "SUPPORTED") if verification else 0,
            contradicted_claims=sum(1 for c in verification.get("claims", []) if c.get("status") == "CONTRADICTED") if verification else 0,
            unverifiable_claims=sum(1 for c in verification.get("claims", []) if c.get("status") not in ["SUPPORTED", "CONTRADICTED"]) if verification else 0,
            revision_count=ethics.get("revision_count", 0) if ethics else 0,
            hallucination_impactful=getattr(result, "hallucination_impactful", False),
            hallucination_detected=getattr(result, "hallucination_detected", False),
            abstained=result.abstained,
            suppressed=result.suppressed,
            agent_traces=result.trace,
            verification_trace=verification,
            consensus_trace=consensus,
            ethics_trace=ethics,
        )

        try:
            scored = evaluator.score(result, case.get("ground_truth", ""))
            telemetry.correctness = scored.top1_correct
            telemetry.top3_correct = scored.top3_correct
            telemetry.confidence = result.output_confidence or 0.0
            telemetry.consensus = not result.abstained
            telemetry.abstained = result.abstained
            telemetry.suppressed = result.suppressed
            telemetry.human_review_required = result.abstained or result.suppressed
            telemetry.hallucination_impactful = scored.hallucination_impactful
            telemetry.unsafe_reassurance = scored.unsafe_reassurance
        except Exception:
            pass

        return telemetry