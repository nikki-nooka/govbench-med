"""
Phase 3: Research Analysis Module for GovBench-Med.
Performs non-parametric statistical testing (Wilcoxon signed-rank test), 95% CIs,
Pareto frontier calculation, marginal overhead analysis, and error taxonomy.
"""

import math
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Tuple


class ResearchAnalyzer:
    """
    Comprehensive research analysis engine for GovBench-Med experiment telemetry.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def g0_to_g4_comparison(self) -> pd.DataFrame:
        """Aggregate G0-G4 metrics per governance level."""
        agg = self.df.groupby("governance_level").agg(
            n=("case_id", "count"),
            accuracy=("correctness", "mean"),
            top3_accuracy=("top3_correct", "mean"),
            cmr=("critical_miss", "mean"),
            hir=("hallucination_impactful", "mean"),
            urr=("unsafe_reassurance", "mean"),
            abstention_rate=("abstained", "mean"),
            mean_tokens=("total_tokens", "mean"),
            mean_latency_ms=("latency_ms", "mean"),
        ).reset_index()

        # Compute CSS
        agg["css"] = 1.0 - (0.50 * agg["cmr"] + 0.30 * agg["hir"] + 0.20 * agg["urr"])

        # Compute CCS
        g0_tokens = agg[agg["governance_level"] == "G0"]["mean_tokens"].values[0] if "G0" in agg["governance_level"].values else agg["mean_tokens"].min()
        g0_latency = agg[agg["governance_level"] == "G0"]["mean_latency_ms"].values[0] if "G0" in agg["governance_level"].values else agg["mean_latency_ms"].min()

        agg["tc_norm"] = agg["mean_tokens"] / max(g0_tokens, 1.0)
        agg["lat_norm"] = agg["mean_latency_ms"] / max(g0_latency, 1.0)
        agg["ccs"] = 0.60 * agg["tc_norm"] + 0.40 * agg["lat_norm"]

        return agg

    def marginal_overhead_analysis(self) -> pd.DataFrame:
        """Compute marginal quality gain, token overhead, latency overhead, and GE."""
        agg = self.g0_to_g4_comparison().set_index("governance_level")
        order = [l for l in ["G0", "G1", "G2", "G3", "G4"] if l in agg.index]

        rows = []
        for i in range(len(order) - 1):
            l0, l1 = order[i], order[i + 1]
            row0 = agg.loc[l0]
            row1 = agg.loc[l1]

            d_css = row1["css"] - row0["css"]
            d_acc = row1["accuracy"] - row0["accuracy"]
            d_tokens = row1["mean_tokens"] - row0["mean_tokens"]
            d_latency = row1["mean_latency_ms"] - row0["mean_latency_ms"]
            d_ccs = row1["ccs"] - row0["ccs"]

            ge = d_css / d_ccs if abs(d_ccs) > 1e-6 else (999.0 if d_css > 0 else 0.0)

            rows.append({
                "transition": f"{l0} -> {l1}",
                "delta_css": round(d_css, 4),
                "delta_accuracy": round(d_acc, 4),
                "delta_tokens": round(d_tokens, 1),
                "delta_latency_ms": round(d_latency, 1),
                "delta_ccs": round(d_ccs, 4),
                "ge": round(ge, 4),
                "knee_of_curve": 0.0 < ge < 1.0,
            })

        return pd.DataFrame(rows)

    def statistical_significance_tests(self) -> pd.DataFrame:
        """
        Perform paired Wilcoxon signed-rank test on correctness and latency across levels.
        """
        levels = [l for l in ["G0", "G1", "G2", "G3", "G4"] if l in self.df["governance_level"].unique()]
        results = []

        for i in range(len(levels) - 1):
            l0, l1 = levels[i], levels[i + 1]
            df0 = self.df[self.df["governance_level"] == l0].sort_values("case_id")
            df1 = self.df[self.df["governance_level"] == l1].sort_values("case_id")

            # Align on case_id
            merged = pd.merge(df0, df1, on="case_id", suffixes=(f"_{l0}", f"_{l1}"))
            if len(merged) < 5:
                continue

            # Accuracy difference test
            try:
                stat_acc, p_acc = stats.wilcoxon(merged[f"correctness_{l0}"], merged[f"correctness_{l1}"])
            except Exception:
                stat_acc, p_acc = 0.0, 1.0

            # Latency difference test
            try:
                stat_lat, p_lat = stats.wilcoxon(merged[f"latency_ms_{l0}"], merged[f"latency_ms_{l1}"])
            except Exception:
                stat_lat, p_lat = 0.0, 1.0

            results.append({
                "comparison": f"{l0} vs {l1}",
                "n_paired": len(merged),
                "acc_p_value": round(p_acc, 5),
                "acc_statistically_significant": p_acc < 0.05,
                "latency_p_value": round(p_lat, 5),
                "latency_statistically_significant": p_lat < 0.05,
            })

        return pd.DataFrame(results)

    def compute_bootstrap_confidence_intervals(self, n_bootstrap: int = 1000, ci: float = 95.0) -> pd.DataFrame:
        """Compute bootstrap 95% CIs for CSS, Accuracy, and Cost across governance levels."""
        levels = self.df["governance_level"].unique()
        rows = []

        alpha = (100.0 - ci) / 2.0

        for lvl in levels:
            sub = self.df[self.df["governance_level"] == lvl]
            n = len(sub)
            if n == 0:
                continue

            css_samples = []
            acc_samples = []

            for _ in range(n_bootstrap):
                sample = sub.sample(n=n, replace=True)
                cmr = sample["critical_miss"].mean() if "critical_miss" in sample else 0
                hir = sample["hallucination_impactful"].mean() if "hallucination_impactful" in sample else 0
                urr = sample["unsafe_reassurance"].mean() if "unsafe_reassurance" in sample else 0
                css = 1.0 - (0.50 * cmr + 0.30 * hir + 0.20 * urr)
                acc = sample["correctness"].mean()
                css_samples.append(css)
                acc_samples.append(acc)

            rows.append({
                "governance_level": lvl,
                "css_mean": round(float(np.mean(css_samples)), 4),
                "css_ci_lower": round(float(np.percentile(css_samples, alpha)), 4),
                "css_ci_upper": round(float(np.percentile(css_samples, 100.0 - alpha)), 4),
                "acc_mean": round(float(np.mean(acc_samples)), 4),
                "acc_ci_lower": round(float(np.percentile(acc_samples, alpha)), 4),
                "acc_ci_upper": round(float(np.percentile(acc_samples, 100.0 - alpha)), 4),
            })

        return pd.DataFrame(rows)
