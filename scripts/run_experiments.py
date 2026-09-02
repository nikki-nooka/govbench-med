"""
Main experiment runner for GovBench-Med.
Runs G0-G4 governance levels independently on shared case datasets.
Logs complete telemetry and outputs aggregate metrics + figures.

Usage:
    python scripts/run_experiments.py --n 100         # 100-case validated benchmark
    python scripts/run_experiments.py --n 50          # 50-case pilot
    python scripts/run_experiments.py --full          # Full dataset
"""

import argparse
import json
import csv
import sys
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.governance.levels import run_governance, GovernanceResult
from src.evaluation.metrics import CaseEvaluator, ScoredResult, compute_aggregate_metrics, compute_governance_efficiency
from src.evaluation.telemetry import RunTelemetry
from src.evaluation.analysis import ResearchAnalyzer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS = [
    "llama3.1:8b",       # Meta Llama 3.1 8B
    "mistral:7b",        # Mistral 7B
    "qwen2.5:7b",        # Qwen 2.5 7B
]

GOVERNANCE_LEVELS = ["G0", "G1", "G2", "G3", "G4"]
SEEDS = [42]

DATA_PATH = ROOT / "data" / "processed" / "cases.json"
RESULTS_DIR = ROOT / "experiments" / "results"
LOGS_DIR = ROOT / "experiments" / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(case: dict, model: str, level: str, seed: int, evaluator: CaseEvaluator) -> tuple[ScoredResult, RunTelemetry]:
    """
    Run a single (case, model, level, seed) combination.
    Measures REAL wall-clock latency in milliseconds.
    """
    start_time = time.perf_counter()
    
    try:
        gov_result = run_governance(level, case, model, seed)
    except Exception as e:
        print(f"  ERROR: {case['id']} {model} {level} seed={seed}: {e}")
        gov_result = GovernanceResult(
            case_id=case["id"], governance_level=level,
            model=model, seed=seed, prompt_version="v1.1",
            abstained=True,
        )

    # Calculate real total elapsed latency in MS
    total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    if gov_result.total_latency <= 0:
        gov_result.total_latency = total_elapsed_ms / 1000.0

    # Score with robust diagnosis matcher
    ground_truth = case.get("ground_truth", "")
    ground_truth_key = case.get("ground_truth_key", "")
    options = case.get("options", {})
    
    scored = evaluator.score(gov_result, ground_truth, ground_truth_key, options)

    # Build telemetry object
    telemetry = RunTelemetry.from_governance_result(gov_result, case, model, level, seed, evaluator)
    telemetry.latency_ms = max(total_elapsed_ms, gov_result.total_latency * 1000.0)
    
    # Save trace log
    log_path = LOGS_DIR / f"{case['id']}_{level}_{model.replace(':', '_')}_seed{seed}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(telemetry.to_dict(), f, indent=2)

    return scored, telemetry


def run_matrix(cases: list, models: list, levels: list, seeds: list,
               evaluator: CaseEvaluator) -> tuple[list[ScoredResult], list[RunTelemetry]]:
    """Run full benchmark matrix."""
    total = len(cases) * len(models) * len(levels) * len(seeds)
    print(f"\n=======================================================")
    print(f"GovBench-Med Benchmark Execution")
    print(f"Cases: {len(cases)} | Models: {len(models)} | Levels: {len(levels)} | Seeds: {len(seeds)}")
    print(f"Total Runs: {total}")
    print(f"=======================================================\n")

    tasks = [
        (case, model, level, seed)
        for case in cases
        for model in models
        for level in levels
        for seed in seeds
    ]

    all_scored = []
    all_telemetry = []
    done = 0
    t_start = time.time()

    for case, model, level, seed in tasks:
        print(f"  [{done+1}/{total}] {case['id']} ({case.get('source','').upper()}) | {model} | {level} | seed={seed}", end=" ... ", flush=True)
        
        scored, telemetry = run_one(case, model, level, seed, evaluator)
        
        status = "ABSTAIN" if scored.abstained else ("✓ MATCH" if scored.top1_correct else "✗ MISS")
        print(f"{status} | {telemetry.total_tokens} tok | {telemetry.latency_ms:.1f} ms")
        
        all_scored.append(scored)
        all_telemetry.append(telemetry)
        done += 1

    total_elapsed = time.time() - t_start
    print(f"\nDone. {total} runs in {total_elapsed/60:.2f} min.")
    return all_scored, all_telemetry


def save_results(telemetry_list: list[RunTelemetry], tag: str) -> Path:
    """Save flat telemetry CSV."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"results_{tag}_{ts}.csv"

    if not telemetry_list:
        return csv_path

    fieldnames = list(telemetry_list[0].to_csv_row().keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in telemetry_list:
            writer.writerow(t.to_csv_row())

    print(f"Telemetry CSV saved to: {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="GovBench-Med Benchmark Execution Engine")
    parser.add_argument("--n", type=int, default=100, help="Number of cases to benchmark (default: 100)")
    parser.add_argument("--pilot", action="store_true", help="Run 50-case pilot")
    parser.add_argument("--full", action="store_true", help="Run full dataset")
    parser.add_argument("--level", type=str, default=None, help="Single governance level override")
    parser.add_argument("--model", type=str, default=None, help="Single model override")
    args = parser.parse_args()

    if not DATA_PATH.exists():
        print(f"ERROR: Cases file not found at {DATA_PATH}")
        sys.exit(1)

    with open(DATA_PATH, encoding="utf-8") as f:
        all_cases = json.load(f)

    if args.pilot:
        n_cases = 50
        tag = "pilot_50"
    elif args.full:
        n_cases = len(all_cases)
        tag = "full"
    else:
        n_cases = args.n
        tag = f"benchmark_{n_cases}"

    cases = all_cases[:n_cases]
    models = [args.model] if args.model else [MODELS[0]]
    levels = [args.level] if args.level else GOVERNANCE_LEVELS
    seeds = SEEDS

    high_acuity_ids = {c["id"] for c in all_cases if c.get("is_high_acuity")}
    ambiguous_ids = {c["id"] for c in all_cases if c.get("is_ambiguous")}
    
    evaluator = CaseEvaluator(high_acuity_ids, ambiguous_ids)

    scored_list, telemetry_list = run_matrix(cases, models, levels, seeds, evaluator)

    csv_path = save_results(telemetry_list, tag)

    # Perform Research Analysis
    import pandas as pd
    df = pd.DataFrame([t.to_csv_row() for t in telemetry_list])
    analyzer = ResearchAnalyzer(df)
    
    comp_df = analyzer.g0_to_g4_comparison()
    marg_df = analyzer.marginal_overhead_analysis()

    print("\n==========================================================================================")
    print("GOVERNANCE-COST BENCHMARK SUMMARY (G0 -> G4)")
    print("==========================================================================================")
    print(comp_df[["governance_level", "n", "accuracy", "top3_accuracy", "cmr", "hir", "urr", "css", "mean_tokens", "mean_latency_ms", "ccs"]].to_string(index=False))

    print("\n==========================================================================================")
    print("MARGINAL OVERHEAD & GOVERNANCE EFFICIENCY (GE)")
    print("==========================================================================================")
    print(marg_df.to_string(index=False))

    # Save summary JSON
    summary_path = RESULTS_DIR / f"summary_{tag}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "comparison": comp_df.to_dict(orient="records"),
            "marginal": marg_df.to_dict(orient="records"),
        }, f, indent=2)
    print(f"\nSummary JSON saved to: {summary_path}")


if __name__ == "__main__":
    main()
