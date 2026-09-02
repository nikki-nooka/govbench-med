"""
Fetch real MedQA-USMLE data from HuggingFace datasets viewer API.
No pyarrow or HuggingFace library needed — pure stdlib urllib.
"""
import json
import random
import re
import sys
import time
from pathlib import Path
import urllib.request
import urllib.error

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

HF_API = "https://datasets-server.huggingface.co/rows"
MEDQA_DATASET = "GBaker/MedQA-USMLE-4-options"
BATCH_SIZE = 100   # HF API max per request

HIGH_ACUITY_KEYWORDS = [
    "myocardial infarction", "heart attack", "stemi", "nstemi",
    "pulmonary embolism", "sepsis", "septic shock",
    "stroke", "cerebrovascular accident", "cva", "tia",
    "subarachnoid hemorrhage", "meningitis", "encephalitis",
    "aortic dissection", "tension pneumothorax", "anaphylaxis",
    "diabetic ketoacidosis", "dka", "hypertensive emergency",
    "acute liver failure", "ectopic pregnancy", "bowel perforation",
    "inferior stemi", "cardiac arrest", "respiratory failure",
]

def fetch_medqa(n=200, seed=42) -> list[dict]:
    cache = RAW_DIR / "medqa_real.json"
    if cache.exists():
        print(f"  Using cached MedQA: {cache}")
        with open(cache) as f:
            rows = json.load(f)
        print(f"  Loaded {len(rows)} cached rows.")
    else:
        rows = []
        offset = 0
        print(f"  Fetching MedQA from HuggingFace API (need {n} rows)...")
        while len(rows) < n + 50:  # fetch extra for shuffle
            url = (
                f"{HF_API}?dataset={urllib.request.quote(MEDQA_DATASET)}"
                f"&config=default&split=test&offset={offset}&length={BATCH_SIZE}"
            )
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = json.loads(r.read())
                batch = [item["row"] for item in data.get("rows", [])]
                if not batch:
                    break
                rows.extend(batch)
                offset += len(batch)
                print(f"    Fetched {len(rows)} rows...", end="\r")
                time.sleep(0.3)   # be polite to HF API
            except Exception as e:
                print(f"\n  Warning: fetch stopped at offset {offset}: {e}")
                break

        print(f"\n  Fetched {len(rows)} total rows.")
        with open(cache, "w") as f:
            json.dump(rows, f)

    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]

    cases = []
    for i, row in enumerate(rows):
        options = row.get("options", {})
        answer_key = row.get("answer_idx", "A")
        if isinstance(answer_key, int):
            answer_key = ["A", "B", "C", "D"][min(answer_key, 3)]
        ground_truth = options.get(answer_key, list(options.values())[0] if options else "Unknown")
        question = row.get("question", "")
        age, sex = _extract_demographics(question)

        cases.append({
            "id": f"medqa_{i:04d}",
            "source": "medqa",
            "age": age, "sex": sex,
            "chief_complaint": "", "history": "", "symptoms": [],
            "question": question,
            "options": options,
            "ground_truth": ground_truth,
            "ground_truth_key": answer_key,
            "is_high_acuity": False,
            "is_ambiguous": False,
            "severity_tier": 3,
        })

    print(f"  Prepared {len(cases)} MedQA cases.")
    return cases


def fetch_ddxplus_conditions() -> tuple[dict, dict]:
    """Fetch DDXPlus condition + evidence metadata from HF API."""
    cache_c = RAW_DIR / "ddxplus_conditions.json"
    cache_e = RAW_DIR / "ddxplus_evidences.json"

    # Try HF datasets viewer for metadata files
    conditions, evidences = {}, {}

    # DDXPlus metadata is in the repo files — try the HF file API
    for fname, cache, out in [
        ("release_conditions.json", cache_c, conditions),
        ("release_evidences.json",  cache_e, evidences),
    ]:
        if cache.exists():
            with open(cache) as f:
                out.update(json.load(f))
            print(f"  Using cached {fname}")
            continue

        url = f"https://huggingface.co/datasets/mila-iqia/ddxplus/resolve/main/{fname}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
            out.update(data)
            with open(cache, "w") as f:
                json.dump(data, f)
            print(f"  Downloaded {fname} ({len(data)} entries)")
        except Exception as e:
            print(f"  Could not fetch {fname}: {e}")

    return conditions, evidences


def build_ddxplus(n=100, seed=42) -> list[dict]:
    """Fetch DDXPlus test rows via HF API, build cases."""
    DDX_DATASET = "aai530-group6/ddxplus"
    cache = RAW_DIR / "ddxplus_real.json"

    conditions, evidences = fetch_ddxplus_conditions()

    if cache.exists():
        print(f"  Using cached DDXPlus: {cache}")
        with open(cache) as f:
            rows = json.load(f)
    else:
        rows = []
        offset = 0
        print(f"  Fetching DDXPlus from HuggingFace API...")
        while len(rows) < n + 50:
            url = (
                f"{HF_API}?dataset={urllib.request.quote(DDX_DATASET)}"
                f"&config=default&split=test&offset={offset}&length={BATCH_SIZE}"
            )
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = json.loads(r.read())
                batch = [item["row"] for item in data.get("rows", [])]
                if not batch:
                    break
                rows.extend(batch)
                offset += len(batch)
                print(f"    Fetched {len(rows)} rows...", end="\r")
                time.sleep(0.3)
            except Exception as e:
                print(f"\n  Warning: fetch stopped: {e}")
                break

        print(f"\n  Fetched {len(rows)} DDXPlus rows.")
        with open(cache, "w") as f:
            json.dump(rows, f)

    rng = random.Random(seed)
    rng.shuffle(rows)

    cases = []
    for i, row in enumerate(rows):
        if len(cases) >= n:
            break
        try:
            diff_raw = row.get("DIFFERENTIAL_DIAGNOSIS", [])
            if isinstance(diff_raw, str):
                diff_raw = json.loads(diff_raw)
        except Exception:
            diff_raw = []

        ground_truth = row.get("PATHOLOGY", "Unknown")
        # Resolve condition name if we have metadata
        if conditions and ground_truth in conditions:
            ground_truth = conditions[ground_truth].get("condition_name", ground_truth)

        ev_raw = row.get("EVIDENCES", [])
        if isinstance(ev_raw, str):
            try:
                ev_raw = json.loads(ev_raw)
            except Exception:
                ev_raw = []

        symptoms = []
        for ev_code in ev_raw[:8]:
            if evidences and str(ev_code) in evidences:
                symptoms.append(evidences[str(ev_code)].get("question_en", str(ev_code)))
            else:
                symptoms.append(str(ev_code))

        # Build differential
        differential = []
        if diff_raw:
            for item in (diff_raw if isinstance(diff_raw[0], (list, tuple)) else [[d, 0.5] for d in diff_raw]):
                dname = item[0] if isinstance(item, (list, tuple)) else item
                dprob = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else 0.5
                if conditions and dname in conditions:
                    dname = conditions[dname].get("condition_name", dname)
                differential.append({"diagnosis": dname, "probability": float(dprob)})

        cases.append({
            "id": f"ddxplus_{i:04d}",
            "source": "ddxplus",
            "age": str(row.get("AGE", "unknown")),
            "sex": str(row.get("SEX", "unknown")),
            "chief_complaint": symptoms[0] if symptoms else "",
            "history": "",
            "symptoms": symptoms,
            "question": "What is the most likely diagnosis for this patient?",
            "options": {},
            "ground_truth": ground_truth,
            "ground_truth_key": "",
            "is_high_acuity": False,
            "is_ambiguous": len(differential) >= 3,
            "severity_tier": 3,
            "differential": differential,
        })

    print(f"  Prepared {len(cases)} DDXPlus cases.")
    return cases


def tag_high_acuity(cases: list[dict]) -> list[dict]:
    count = 0
    for case in cases:
        text = (case.get("ground_truth","") + " " + case.get("question","")).lower()
        for kw in HIGH_ACUITY_KEYWORDS:
            if kw in text:
                case["is_high_acuity"] = True
                case["severity_tier"] = 1
                count += 1
                break
    print(f"  Tagged {count} high-acuity cases.")
    return cases


def _extract_demographics(question: str):
    age, sex = "unknown", "unknown"
    m = re.search(r"(\d+)[- ]year[- ]old", question, re.IGNORECASE)
    if m:
        age = m.group(1)
    if re.search(r"\bwoman\b|\bfemale\b|\bgirl\b", question, re.IGNORECASE):
        sex = "female"
    elif re.search(r"\bman\b|\bmale\b|\bboy\b", question, re.IGNORECASE):
        sex = "male"
    return age, sex


def main():
    print("=== GovBench-Med Dataset Preparation (real data) ===\n")

    print("[1/3] Fetching MedQA-USMLE (200 real cases)...")
    medqa = fetch_medqa(n=200, seed=42)

    print("\n[2/3] Fetching DDXPlus (100 real cases)...")
    ddxplus = build_ddxplus(n=100, seed=42)

    print("\n[3/3] Merging and tagging...")
    all_cases = medqa + ddxplus
    all_cases = tag_high_acuity(all_cases)

    out = PROCESSED_DIR / "cases.json"
    with open(out, "w") as f:
        json.dump(all_cases, f, indent=2)

    sources = {}
    for c in all_cases:
        sources[c["source"]] = sources.get(c["source"], 0) + 1
    ha  = sum(c["is_high_acuity"] for c in all_cases)
    amb = sum(c["is_ambiguous"]   for c in all_cases)

    print(f"\n=== Dataset Summary ===")
    print(f"Total cases : {len(all_cases)}")
    for s, cnt in sources.items():
        print(f"  {s:10}: {cnt}")
    print(f"High-acuity : {ha}  ({100*ha/len(all_cases):.1f}%)")
    print(f"Ambiguous   : {amb} ({100*amb/len(all_cases):.1f}%)")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
