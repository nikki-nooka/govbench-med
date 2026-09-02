"""
Load and preprocess cases from MedQA-USMLE and DDXPlus.
Outputs a unified JSON format understood by all governance levels.

Unified case schema:
{
    "id": str,
    "source": "medqa" | "ddxplus",
    "age": str,
    "sex": str,
    "chief_complaint": str,
    "history": str,
    "symptoms": list[str],
    "question": str,           # MedQA only
    "options": dict,           # MedQA only — {"A": "...", "B": "...", ...}
    "ground_truth": str,       # diagnosis label
    "ground_truth_key": str,   # MedQA option key (A/B/C/D)
    "is_high_acuity": bool,    # manually set via HIGH_ACUITY_IDS
    "is_ambiguous": bool,      # True if DDXPlus differential has ≥3 entries
    "severity_tier": int,      # 1=life-threatening, 2=urgent, 3=routine (manual)
}
"""

import json
import random
from pathlib import Path
from typing import Optional


# High-acuity case IDs — manually reviewed, life-threatening diagnoses
# Populated after inspecting MedQA answer distribution
HIGH_ACUITY_IDS: set[str] = set()   # filled in scripts/tag_high_acuity.py

# ---------------------------------------------------------------------------
# MedQA loader
# ---------------------------------------------------------------------------

def load_medqa(jsonl_path: str, n: int = 300, seed: int = 42) -> list[dict]:
    """
    Load n cases from MedQA-USMLE jsonl file.
    MedQA format: {"question": "...", "options": {"A": ..., "B": ...}, "answer": "A", "answer_idx": "A", "metamap_phrases": [...]}

    Download from: https://github.com/jind11/MedQA
    File: data_clean/questions/US/4_options/phrases_no_exclude_test.jsonl
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(
            f"MedQA file not found: {jsonl_path}\n"
            "Download from: https://github.com/jind11/MedQA\n"
            "File: data_clean/questions/US/4_options/phrases_no_exclude_test.jsonl"
        )

    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            cases.append(raw)

    # Reproducible sample
    rng = random.Random(seed)
    rng.shuffle(cases)
    cases = cases[:n]

    processed = []
    for i, raw in enumerate(cases):
        case_id = f"medqa_{i:04d}"
        answer_key = raw.get("answer_idx") or raw.get("answer", "A")
        ground_truth = raw.get("options", {}).get(answer_key, "Unknown")

        # Extract patient demographics from question text (best-effort)
        question = raw.get("question", "")
        age, sex = _extract_demographics(question)

        processed.append({
            "id": case_id,
            "source": "medqa",
            "age": age,
            "sex": sex,
            "chief_complaint": "",
            "history": "",
            "symptoms": [],
            "question": question,
            "options": raw.get("options", {}),
            "ground_truth": ground_truth,
            "ground_truth_key": answer_key,
            "is_high_acuity": case_id in HIGH_ACUITY_IDS,
            "is_ambiguous": False,
            "severity_tier": 3,  # updated by tag_high_acuity.py
        })

    return processed


# ---------------------------------------------------------------------------
# DDXPlus loader
# ---------------------------------------------------------------------------

def load_ddxplus(
    patients_csv: str,
    conditions_json: str,
    evidences_json: str,
    n: int = 100,
    seed: int = 42,
    min_differential_size: int = 1,
) -> list[dict]:
    """
    Load n cases from DDXPlus English dataset.
    Download from: https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585

    patients_csv:    release_test_patients.csv (or train/validate)
    conditions_json: release_conditions.json
    evidences_json:  release_evidences.json
    """
    import csv

    path = Path(patients_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"DDXPlus patients file not found: {patients_csv}\n"
            "Download from: https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585"
        )

    with open(conditions_json, "r") as f:
        conditions = json.load(f)
    with open(evidences_json, "r") as f:
        evidences = json.load(f)

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n * 3]  # oversample, then filter

    processed = []
    for i, row in enumerate(rows):
        if len(processed) >= n:
            break
        try:
            differential_raw = json.loads(row.get("DIFFERENTIAL_DIAGNOSIS", "[]"))
        except Exception:
            continue

        if len(differential_raw) < min_differential_size:
            continue

        ground_truth = row.get("PATHOLOGY", "Unknown")
        cond_info = conditions.get(ground_truth, {})
        cond_name = cond_info.get("condition_name", ground_truth)

        # Parse evidence list
        evidence_list = json.loads(row.get("EVIDENCES", "[]"))
        symptom_names = []
        for ev_code in evidence_list:
            ev_info = evidences.get(ev_code, {})
            ev_name = ev_info.get("question_en", ev_code)
            symptom_names.append(ev_name)

        differential = [
            {"diagnosis": conditions.get(d, {}).get("condition_name", d), "probability": p}
            for d, p in differential_raw
        ]

        is_ambiguous = len(differential) >= 3
        case_id = f"ddxplus_{i:04d}"

        processed.append({
            "id": case_id,
            "source": "ddxplus",
            "age": row.get("AGE", "unknown"),
            "sex": row.get("SEX", "unknown"),
            "chief_complaint": symptom_names[0] if symptom_names else "",
            "history": "",
            "symptoms": symptom_names,
            "question": f"What is the most likely diagnosis for this patient?",
            "options": {},
            "ground_truth": cond_name,
            "ground_truth_key": "",
            "is_high_acuity": case_id in HIGH_ACUITY_IDS,
            "is_ambiguous": is_ambiguous,
            "severity_tier": 3,
            "differential": differential,
        })

    return processed


# ---------------------------------------------------------------------------
# Merge and save
# ---------------------------------------------------------------------------

def build_dataset(
    medqa_path: Optional[str],
    ddxplus_patients: Optional[str],
    ddxplus_conditions: Optional[str],
    ddxplus_evidences: Optional[str],
    output_path: str,
    n_medqa: int = 200,
    n_ddxplus: int = 100,
    seed: int = 42,
):
    cases = []
    if medqa_path:
        print(f"Loading {n_medqa} MedQA cases...")
        cases += load_medqa(medqa_path, n=n_medqa, seed=seed)
    if ddxplus_patients:
        print(f"Loading {n_ddxplus} DDXPlus cases...")
        cases += load_ddxplus(ddxplus_patients, ddxplus_conditions, ddxplus_evidences,
                              n=n_ddxplus, seed=seed)

    print(f"Total cases: {len(cases)}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)
    print(f"Saved to {output_path}")
    return cases


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _extract_demographics(question: str):
    """Best-effort extraction of age and sex from MedQA question text."""
    import re
    age = "unknown"
    sex = "unknown"

    age_match = re.search(r"(\d+)[- ]year[- ]old", question, re.IGNORECASE)
    if age_match:
        age = age_match.group(1)

    if re.search(r"\bwoman\b|\bfemale\b|\bgirl\b", question, re.IGNORECASE):
        sex = "female"
    elif re.search(r"\bman\b|\bmale\b|\bboy\b", question, re.IGNORECASE):
        sex = "male"

    return age, sex
