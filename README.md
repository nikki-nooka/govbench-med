# GovBench-Med
### Quantifying the Governance-Cost Tradeoff in Multi-Agent Clinical Diagnosis Systems

**B.Tech Final Year Project | IEEE Access Target Submission**

---

## What This Is
A benchmark study that treats AI governance as a *tunable dial* — not an on/off switch — and
measures exactly how much clinical safety each level of oversight costs in computation.

## Project Structure
```
govbench-med/
├── src/
│   ├── agents/          # Agent definitions (diagnostician, critic, verifier, etc.)
│   ├── governance/      # G0–G4 governance level implementations
│   ├── evaluation/      # Safety + cost metrics
│   └── utils/           # Token counting, logging, helpers
├── data/
│   ├── raw/             # Downloaded MedQA + DDXPlus files
│   └── processed/       # Cleaned, formatted cases (JSON)
├── experiments/
│   ├── results/         # CSV outputs from each run
│   └── logs/            # Per-case trace logs
├── paper/
│   ├── figures/         # Pareto curves, heatmaps
│   └── tables/          # LaTeX tables
└── scripts/             # Data prep, run scripts, analysis
```

## Timeline
| Week | Phase | Goal |
|------|-------|------|
| 1–2  | Foundation | Spec, metrics, repo, env |
| 3–4  | Implementation | All 5 governance levels coded |
| 5–6  | Experiments | Full matrix run + analysis |
| 7–8  | Paper | Write + submit to IEEE Access |

## Team
- Add your team members here

## Key Papers to Cite
- TeamMedAgents (2025) — Pareto efficiency in multi-agent medical reasoning
- ConfAgents (2025) — Cost-adaptive multi-agent clinical diagnosis
- MedSafe-Dx (2026) — Safety-focused diagnostic benchmark
- MDAgents (NeurIPS 2024) — Adaptive multi-agent medical decision making
- MedAgents (2023) — Role-playing specialist agents for medical QA
