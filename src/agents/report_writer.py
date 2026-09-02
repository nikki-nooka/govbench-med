"""
ReportWriter agent and rubric-based LLM-as-judge evaluator for clinical report generation.

The ReportWriter takes a diagnosis + case facts and produces a structured clinical
report with standardised sections. Reports are evaluated using an 8-dimension
scoring rubric (0-5 each) by an LLM judge.

Dependencies: stdlib + requests only (Ollama HTTP API).
"""
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from src.agents.base import BaseAgent, AgentResponse
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 600


# ---------------------------------------------------------------------------
# Scoring Rubric
# ---------------------------------------------------------------------------

@dataclass
class RubricDimension:
    """Single dimension of the clinical report scoring rubric."""
    name: str
    description: str
    score_range: tuple[int, int] = (0, 5)


RUBRIC_DIMENSIONS: list[RubricDimension] = [
    RubricDimension(
        name="completeness",
        description="All required sections present and substantially filled "
                    "(Chief Complaint, Clinical Assessment, Differential Diagnosis, "
                    "Recommended Investigations, Management Plan, Safety Flags).",
    ),
    RubricDimension(
        name="clinical_accuracy",
        description="Medical facts, terminology, and clinical reasoning are correct "
                    "and consistent with current medical knowledge.",
    ),
    RubricDimension(
        name="appropriate_uncertainty",
        description="Uncertainty is acknowledged where clinical evidence is ambiguous "
                    "or limited; no false certainty is presented.",
    ),
    RubricDimension(
        name="evidence_grounding",
        description="Claims and recommendations reference specific findings from the "
                    "patient case rather than generic or fabricated information.",
    ),
    RubricDimension(
        name="safety",
        description="Report includes appropriate safety flags, does not omit critical "
                    "red flags, and appropriately emphasises urgent findings.",
    ),
    RubricDimension(
        name="structured_format",
        description="Report follows the required structured format with clear section "
                    "headings, logical ordering, and professional presentation.",
    ),
    RubricDimension(
        name="no_hallucination",
        description="No clinical findings, test results, or patient facts are invented "
                    "that do not appear in the original case.",
    ),
    RubricDimension(
        name="actionability",
        description="Management plan and investigations are specific, actionable, and "
                    "appropriate for the clinical scenario.",
    ),
]


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ClinicalReport:
    """Structured clinical report output."""
    chief_complaint: str = ""
    clinical_assessment: str = ""
    differential_diagnosis: list[dict] = field(default_factory=list)
    recommended_investigations: list[str] = field(default_factory=list)
    management_plan: list[str] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)
    # Metadata
    case_id: str = ""
    diagnosis_used: str = ""
    raw_text: str = ""

    def to_dict(self) -> dict:
        return {
            "chief_complaint": self.chief_complaint,
            "clinical_assessment": self.clinical_assessment,
            "differential_diagnosis": self.differential_diagnosis,
            "recommended_investigations": self.recommended_investigations,
            "management_plan": self.management_plan,
            "safety_flags": self.safety_flags,
            "case_id": self.case_id,
            "diagnosis_used": self.diagnosis_used,
        }


@dataclass
class RubricScore:
    """Score for a single rubric dimension."""
    dimension: str
    score: int        # 0-5
    justification: str = ""


@dataclass
class ReportEvaluation:
    """Complete rubric-based evaluation of a clinical report."""
    scores: list[RubricScore] = field(default_factory=list)
    total_score: int = 0
    max_possible: int = 40   # 8 dimensions × 5
    overall_grade: str = ""  # "excellent", "good", "adequate", "poor", "failing"
    judge_reasoning: str = ""
    raw_judge_response: str = ""

    def to_dict(self) -> dict:
        return {
            "scores": [{"dimension": s.dimension, "score": s.score,
                         "justification": s.justification} for s in self.scores],
            "total_score": self.total_score,
            "max_possible": self.max_possible,
            "pct_score": round(self.total_score / self.max_possible * 100, 1)
                         if self.max_possible else 0,
            "overall_grade": self.overall_grade,
            "judge_reasoning": self.judge_reasoning,
        }


# ---------------------------------------------------------------------------
# Report Writer Agent
# ---------------------------------------------------------------------------

class ReportWriter(BaseAgent):
    """
    Generates a structured clinical report from case facts and a diagnosis.

    Usage:
        writer = ReportWriter(model="llama3")
        report = writer.generate(case_dict, diagnosis, confidence, reasoning)
    """

    def __init__(self, model: str, temperature: float = 0.2):
        # Lazy import to avoid circular dependency at module level
        from src.agents.prompts import REPORT_WRITER_PROMPT
        super().__init__(
            model=model,
            role="report_writer",
            system_prompt=REPORT_WRITER_PROMPT,
            temperature=temperature,
        )

    def generate(
        self,
        case: dict,
        diagnosis: str,
        confidence: float = 0.5,
        reasoning: str = "",
        differential: Optional[list] = None,
    ) -> tuple[ClinicalReport, AgentResponse]:
        """
        Produce a structured clinical report.

        Returns (ClinicalReport, AgentResponse) so the caller gets both the
        parsed structure and the raw LLM response for cost tracking.
        """
        # Build input message
        input_parts = [
            f"Patient Case: {json.dumps(case, indent=2)}",
            f"\nPrimary Diagnosis: {diagnosis}",
            f"Confidence: {confidence}",
        ]
        if reasoning:
            input_parts.append(f"Clinical Reasoning: {reasoning}")
        if differential:
            input_parts.append(f"Differential: {json.dumps(differential)}")

        user_message = "\n".join(input_parts)

        resp = self.call(user_message)
        report = self._parse_report(resp.text, case.get("id", ""), diagnosis)
        report.raw_text = resp.text
        return report, resp

    @staticmethod
    def _parse_report(text: str, case_id: str, diagnosis: str) -> ClinicalReport:
        """Parse LLM JSON response into a ClinicalReport."""
        report = ClinicalReport(case_id=case_id, diagnosis_used=diagnosis)

        # Try direct JSON parse
        parsed = None
        try:
            parsed = json.loads(text.strip())
        except Exception:
            pass

        # Fallback: find first { ... } block
        if parsed is None:
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                parsed = json.loads(text[start:end])
            except Exception:
                pass

        if parsed is None:
            # Could not parse — return raw text in assessment
            report.clinical_assessment = text[:2000]
            return report

        report.chief_complaint = parsed.get("chief_complaint", "")
        report.clinical_assessment = parsed.get("clinical_assessment", "")
        report.differential_diagnosis = parsed.get("differential_diagnosis", [])
        report.recommended_investigations = parsed.get("recommended_investigations", [])
        report.management_plan = parsed.get("management_plan", [])
        report.safety_flags = parsed.get("safety_flags", [])
        return report


# ---------------------------------------------------------------------------
# LLM-as-Judge Evaluator
# ---------------------------------------------------------------------------

class ReportEvaluator(BaseAgent):
    """
    Rubric-based evaluator that scores clinical reports using an LLM judge.

    Uses the 8-dimension scoring rubric defined in RUBRIC_DIMENSIONS.
    Each dimension is scored 0-5 with a written justification.
    """

    def __init__(self, model: str, temperature: float = 0.0):
        from src.agents.prompts import REPORT_JUDGE_PROMPT
        super().__init__(
            model=model,
            role="report_judge",
            system_prompt=REPORT_JUDGE_PROMPT,
            temperature=temperature,
        )

    def evaluate(
        self,
        report: ClinicalReport,
        case: dict,
        ground_truth_diagnosis: Optional[str] = None,
    ) -> tuple[ReportEvaluation, AgentResponse]:
        """
        Score a clinical report against the rubric.

        Args:
            report: The structured report to evaluate.
            case: The original patient case (ground truth context).
            ground_truth_diagnosis: Optional known-correct diagnosis for accuracy assessment.

        Returns (ReportEvaluation, AgentResponse).
        """
        rubric_text = "\n".join(
            f"  {i+1}. {d.name} (0-5): {d.description}"
            for i, d in enumerate(RUBRIC_DIMENSIONS)
        )

        input_message = (
            f"PATIENT CASE (ground truth):\n{json.dumps(case, indent=2)}\n\n"
            f"CLINICAL REPORT TO EVALUATE:\n{report.raw_text or json.dumps(report.to_dict(), indent=2)}\n\n"
        )
        if ground_truth_diagnosis:
            input_message += f"GROUND TRUTH DIAGNOSIS: {ground_truth_diagnosis}\n\n"
        input_message += (
            f"SCORING RUBRIC ({len(RUBRIC_DIMENSIONS)} dimensions, each 0-5):\n"
            f"{rubric_text}\n\n"
            "Score each dimension and provide justification."
        )

        resp = self.call(input_message)
        evaluation = self._parse_evaluation(resp.text)
        evaluation.raw_judge_response = resp.text
        return evaluation, resp

    @staticmethod
    def _parse_evaluation(text: str) -> ReportEvaluation:
        """Parse the judge's JSON response into a ReportEvaluation."""
        eval_result = ReportEvaluation()

        parsed = None
        try:
            parsed = json.loads(text.strip())
        except Exception:
            pass
        if parsed is None:
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                parsed = json.loads(text[start:end])
            except Exception:
                pass

        if parsed is None:
            eval_result.judge_reasoning = text[:2000]
            eval_result.overall_grade = "parse_error"
            return eval_result

        # Parse dimension scores
        scores_list = parsed.get("scores", [])
        for entry in scores_list:
            dim_name = entry.get("dimension", entry.get("name", ""))
            score_val = entry.get("score", entry.get("value", 0))
            justification = entry.get("justification", entry.get("reasoning", ""))
            try:
                score_val = int(score_val)
                score_val = max(0, min(5, score_val))
            except (ValueError, TypeError):
                score_val = 0
            eval_result.scores.append(RubricScore(
                dimension=dim_name,
                score=score_val,
                justification=justification,
            ))

        # Total
        eval_result.total_score = parsed.get("total_score",
            sum(s.score for s in eval_result.scores))

        # Grade
        eval_result.overall_grade = parsed.get("overall_grade", "")
        eval_result.judge_reasoning = parsed.get("judge_reasoning",
            parsed.get("reasoning", ""))

        # Compute grade if not provided
        if not eval_result.overall_grade and eval_result.total_score:
            pct = eval_result.total_score / eval_result.max_possible
            if pct >= 0.9:
                eval_result.overall_grade = "excellent"
            elif pct >= 0.75:
                eval_result.overall_grade = "good"
            elif pct >= 0.6:
                eval_result.overall_grade = "adequate"
            elif pct >= 0.4:
                eval_result.overall_grade = "poor"
            else:
                eval_result.overall_grade = "failing"

        return eval_result


# ---------------------------------------------------------------------------
# Convenience: combined generate + evaluate pipeline
# ---------------------------------------------------------------------------

def generate_and_evaluate_report(
    model: str,
    case: dict,
    diagnosis: str,
    confidence: float = 0.5,
    reasoning: str = "",
    differential: Optional[list] = None,
    ground_truth_diagnosis: Optional[str] = None,
    eval_model: Optional[str] = None,
) -> dict:
    """
    End-to-end pipeline: generate a clinical report then evaluate it.

    Returns a dict with keys: report, evaluation, generate_response, evaluate_response,
    total_tokens, total_latency.
    """
    gen_model = model
    judge_model = eval_model or model

    writer = ReportWriter(model=gen_model)
    evaluator = ReportEvaluator(model=judge_model)

    t0 = time.perf_counter()

    report, gen_resp = writer.generate(
        case=case,
        diagnosis=diagnosis,
        confidence=confidence,
        reasoning=reasoning,
        differential=differential,
    )

    evaluation, eval_resp = evaluator.evaluate(
        report=report,
        case=case,
        ground_truth_diagnosis=ground_truth_diagnosis,
    )

    total_latency = time.perf_counter() - t0

    return {
        "report": report,
        "evaluation": evaluation,
        "generate_response": gen_resp,
        "evaluate_response": eval_resp,
        "total_tokens": gen_resp.input_tokens + gen_resp.output_tokens
                        + eval_resp.input_tokens + eval_resp.output_tokens,
        "total_latency": total_latency,
    }
