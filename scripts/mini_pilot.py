"""
Minimal overnight run: 20 cases × G0 × llama3.1:8b × seed=42.
Run this in background: python scripts/mini_pilot.py > experiments/results/pilot_log.txt 2>&1
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.governance.levels import run_governance
from src.evaluation.metrics import CaseEvaluator, ScoredResult

with open(ROOT / "data" / "processed" / "cases.json") as f:
    cases = json.load(f)[:20]

ha_ids = {c["id"] for c in cases if c.get("is_high_acuity")}
amb_ids = {c["id"] for c in cases if c.get("is_ambiguous")}
evaluator = CaseEvaluator(ha_ids, amb_ids)

results = []
t_start = time.time()

for i, case in enumerate(cases):
    print(f"[{i+1}/20] {case['id']} ...", flush=True)
    t0 = time.time()
    gov = run_governance("G0", case, "llama3.1:8b", 42)
    scored = evaluator.score(gov, case["ground_truth"])
    results.append(scored.to_dict())
    elapsed = time.time() - t0
    status = "CORRECT" if scored.top1_correct else "WRONG"
    print(f"  {status} | diag={gov.top_diagnosis} | gt={case['ground_truth'][:40]} | {gov.total_tokens}tok | {elapsed:.0f}s")

total = time.time() - t_start
print(f"\nDone in {total/60:.1f} min.")
print(f"Correct: {sum(r['top1_correct'] for r in results)}/20")
print(f"High-acuity misses: {sum(r['critical_miss'] for r in results)}")

out = ROOT / "experiments" / "results" / "pilot_g0_20cases.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {out}")