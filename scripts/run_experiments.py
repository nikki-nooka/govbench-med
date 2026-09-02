"""
Main experiment runner.
Runs the full matrix: 5 governance levels × N models × 3 seeds × 300 cases.

Usage:
    python scripts/run_experiments.py --pilot          # 30 cases, G0+G4 only
    python scripts/run_experiments.py --full           # full 300-case matrix
    python scripts/run_experiments.py --level G2       # single level debug run
"""
import argparse
import json
import csv
import sys
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Make sure src/ is on the path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.governance.levels import run_governance, GovernanceResult
from src.evaluation.metrics import CaseEvaluator, ScoredResult, compute_aggregate_metrics, compute_governance_efficiency

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS = [
    "llama3.1:8b",       # ~4.7 GB — pull with: ollama pull llama3.1:8b
    "mistral:7b",        # ~4.1 GB — pull with: ollama pull mistral:7b
    "qwen2.5:7b",        # ~4.4 GB — pull with: ollama pull qwen2.5:7b
]

GOVERNANCE_LEVELS = ["G0", "G1", "G2", "G3", "G4"]

SEEDS = [42, 123, 777]

DATA_PATH = ROOT / "data" / "processed" / "cases.json"
RESULTS_DIR = ROOT / "experiments" / "results"
LOGS_DIR = ROOT / "experiments" / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(case: dict, model: str, level: str, seed: int, evaluator: CaseEvaluator) -> ScoredResult:
    """Run a single (case, model, level, seed) combination."""
    try:
        gov_result = run_governance(level, case, model, seed)
    except Exception as e:
        print(f"  ERROR: {case['id']} {model} {level} seed={seed}: {e}")
        # Return a placeholder failed result so the run continues
        gov_result = GovernanceResult(
            case_id=case["id"], governance_level=level,
            model=model, seed=seed, prompt_version="v1.0",
            abstained=True,
        )

    # Save trace log
    log_path = LOGS_DIR / f"{case['id']}_{level}_{model.replace(':', '_')}_seed{seed}.json"
    with open(log_path, "w") as f:
        json.dump(gov_result.to_dict(), f, indent=2)

    scored = evaluator.score(gov_result, case["ground_truth"])

    # Inject case-level acuity flags (evaluator uses IDs but case has the flags too)
    scored.is_high_acuity = case.get("is_high_acuity", False)

    return scored


def run_matrix(cases: list, models: list, levels: list, seeds: list,
               evaluator: CaseEvaluator, max_workers: int = 4) -> list[ScoredResult]:
    """
    Run the full experiment matrix.
    Each (case, model, level, seed) combination is one unit of work.
    """
    total = len(cases) * len(models) * len(levels) * len(seeds)
    print(f"\nMatrix size: {len(cases)} cases × {len(models)} models × "
          f"{len(levels)} levels × {len(seeds)} seeds = {total} runs\n")

    tasks = [
        (case, model, level, seed)
        for case in cases
        for model in models
        for level in levels
        for seed in seeds
    ]

    all_results = []
    done = 0
    t_start = time.time()

    # Sequential (safer for Ollama rate limits; switch to ThreadPoolExecutor for speed)
    for case, model, level, seed in tasks:
        print(f"  [{done+1}/{total}] {case['id']} | {model} | {level} | seed={seed}", end=" ... ", flush=True)
        t0 = time.time()
        result = run_one(case, model, level, seed, evaluator)
        elapsed = time.time() - t0
        status = "ABSTAIN" if result.abstained else ("✓" if result.top1_correct else "✗")
        print(f"{status} | {result.total_tokens}tok | {elapsed:.1f}s")
        all_results.append(result)
        done += 1

    total_elapsed = time.time() - t_start
    print(f"\nDone. {total} runs in {total_elapsed/60:.1f} min.")
    return all_results


# ---------------------------------------------------------------------------
# Save + Analyze
# ---------------------------------------------------------------------------

def save_results(results: list[ScoredResult], tag: str):
    """Save flat CSV of all per-case results."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"results_{tag}_{ts}.csv"

    if not results:
        print("No results to save.")
        return csv_path

    fieldnames = list(results[0].to_dict().keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())

    print(f"Results saved to: {csv_path}")
    return csv_path


def analyze_results(results: list[ScoredResult]) -> dict:
    """
    Compute aggregate metrics per (governance_level, model) combination.
    Also compute Governance Efficiency between consecutive levels.
    """
    from collections import defaultdict
    import statistics

    # Group by (model, level)
    groups = defaultdict(list)
    for r in results:
        groups[(r.model, r.governance_level)].append(r)

    # Compute G0 median tokens per model as normalization baseline
    g0_baselines = {}
    for model in {r.model for r in results}:
        g0_results = groups.get((model, "G0"), [])
        if g0_results:
            tokens = [r.total_tokens for r in g0_results]
            g0_baselines[model] = statistics.median(tokens) or 1.0
        else:
            g0_baselines[model] = 1.0

    # Aggregate metrics per group
    agg = {}
    for (model, level), group_results in groups.items():
        baseline = g0_baselines.get(model, 1.0)
        metrics = compute_aggregate_metrics(group_results, baseline)
        agg[(model, level)] = metrics

    # Governance efficiency between consecutive levels
    level_order = ["G0", "G1", "G2", "G3", "G4"]
    for model in {r.model for r in results}:
        for i in range(len(level_order) - 1):
            prev_level = level_order[i]
            next_level = level_order[i + 1]
            m_prev = agg.get((model, prev_level))
            m_next = agg.get((model, next_level))
            if m_prev and m_next:
                ge = compute_governance_efficiency(m_prev, m_next)
                agg[(model, next_level)]["ge_from_prev"] = ge
                print(f"  GE({prev_level}→{next_level}) [{model}]: {ge:.4f}")

    return agg


def print_summary_table(agg: dict):
    """Print a readable summary table to stdout."""
    print("\n" + "="*90)
    print(f"{'Model':<20} {'Level':<6} {'CSS':>6} {'CMR':>6} {'HIR':>6} {'URR':>6} {'Tokens':>8} {'CCS':>6} {'GE':>8}")
    print("-"*90)

    level_order = ["G0", "G1", "G2", "G3", "G4"]
    models = sorted({k[0] for k in agg.keys()})

    for model in models:
        for level in level_order:
            m = agg.get((model, level))
            if not m:
                continue
            ge_str = f"{m.get('ge_from_prev', ''):.4f}" if m.get('ge_from_prev') is not None else "—"
            print(f"{model:<20} {level:<6} {m['css']:>6.3f} {m['cmr']:>6.3f} "
                  f"{m['hir']:>6.3f} {m['urr']:>6.3f} {m['mean_tokens']:>8.0f} "
                  f"{m['ccs']:>6.3f} {ge_str:>8}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GovBench-Med experiment runner")
    parser.add_argument("--pilot", action="store_true", help="30 cases, G0+G4 only")
    parser.add_argument("--full", action="store_true", help="Full 300-case matrix")
    parser.add_argument("--level", type=str, default=None, help="Single governance level")
    parser.add_argument("--model", type=str, default=None, help="Single model override")
    parser.add_argument("--n", type=int, default=None, help="Number of cases to run")
    parser.add_argument("--seed", type=int, default=None, help="Single seed override")
    args = parser.parse_args()

    # Load cases
    if not DATA_PATH.exists():
        print(f"ERROR: Cases file not found at {DATA_PATH}")
        print("Run scripts/prepare_data.py first.")
        sys.exit(1)

    with open(DATA_PATH) as f:
        all_cases = json.load(f)
    print(f"Loaded {len(all_cases)} cases from {DATA_PATH}")

    # Select subset based on mode
    if args.pilot:
        cases = all_cases[:30]
        levels = ["G0", "G4"]
        seeds = [42]
        models = [MODELS[0]]
        tag = "pilot"
    elif args.full:
        cases = all_cases[:300]
        levels = GOVERNANCE_LEVELS
        seeds = SEEDS
        models = MODELS
        tag = "full"
    else:
        cases = all_cases[:(args.n or 30)]
        levels = [args.level] if args.level else GOVERNANCE_LEVELS
        seeds = [args.seed] if args.seed else [42]
        models = [args.model] if args.model else [MODELS[0]]
        tag = "custom"

    # Build evaluator
    high_acuity_ids = {c["id"] for c in all_cases if c.get("is_high_acuity")}
    ambiguous_ids = {c["id"] for c in all_cases if c.get("is_ambiguous")}
    print(f"High-acuity cases: {len(high_acuity_ids)} | Ambiguous: {len(ambiguous_ids)}")
    evaluator = CaseEvaluator(high_acuity_ids, ambiguous_ids)

    # Run
    results = run_matrix(cases, models, levels, seeds, evaluator)

    # Save
    csv_path = save_results(results, tag)

    # Analyze
    agg = analyze_results(results)
    print_summary_table(agg)

    # Save aggregate metrics
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    agg_path = RESULTS_DIR / f"aggregate_{tag}_{ts}.json"
    with open(agg_path, "w") as f:
        json.dump({str(k): v for k, v in agg.items()}, f, indent=2)
    print(f"\nAggregate metrics saved to: {agg_path}")


if __name__ == "__main__":
    main()
