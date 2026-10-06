"""Write results to disk.

* `write_report`  -> aggregate numbers and charts only. Safe to share / commit.
* `save_outputs`  -> model file and engineered cohort. PERSONAL DATA, keep local.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .predict import save_artifact
from .train import TrainingResult

MIN_CELL = 5   # suppress any group smaller than this in shared tables

REPORT_README = """# Rotation Fit — shareable report

Everything in this folder is aggregated: counts, rates, metrics and charts.
It contains no names, NIKs, dates of birth or row-level records, so it can be
shared with reviewers or committed to the repository.

| file | content |
|---|---|
| funnel.csv | inclusion funnel (counts per step) |
| success_by_rating.csv | label balance by pre-rotation rating (groups < {min_cell} suppressed) |
| grid_summary.csv | mean / SD of every model x scaler on the development splits |
| grid_results.csv | every individual fit of the grid (metrics only) |
| holdout.csv | one-time evaluation on the untouched hold-out set |
| calibration.csv | predicted vs observed success rate on the hold-out set |
| significance.csv | likelihood-ratio test per attribute (p-value, Holm-adjusted p, direction) |
| importance.csv | permutation importance with 95 % interval across splits |
| reduced_model.csv / reduced_selection.csv | "significant attributes only" model vs full model |
| subgroups.csv | hold-out AUC by job level / gender (groups >= {min_sub}) |
| feature_summary.csv | distribution summary of every model feature |
| metadata.json | what was trained, on what, and its known limitations |
"""


def _jsonable(o):
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    raise TypeError(type(o))


def _plots(result: TrainingResult, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = result.grid
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    piv = g.pivot_table(index="ratio", columns="model", values="auc", aggfunc="mean")
    err = g.pivot_table(index="ratio", columns="model", values="auc", aggfunc="std")
    piv.plot(kind="bar", yerr=err, ax=axes[0], capsize=3, rot=0)
    axes[0].set_title("ROC-AUC by split ratio (development set)\n(error bars = SD across seeds)")
    axes[0].axhline(0.5, ls="--", c="grey", lw=1)
    axes[0].set_ylabel("ROC-AUC")

    g.pivot_table(index="model", columns="scaler", values="auc", aggfunc="mean") \
        .plot(kind="bar", ax=axes[1], rot=0)
    axes[1].set_title("Standardised vs normalised\n(trees are scale-invariant)")
    axes[1].set_ylabel("ROC-AUC")

    c = result.calibration
    axes[2].plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    axes[2].plot(c["mean_predicted"], c["observed_rate"], marker="o")
    axes[2].set_title(f"Calibration on hold-out — {result.best['model']}\n"
                      "(on the diagonal = trustworthy %)")
    axes[2].set_xlabel("predicted probability")
    axes[2].set_ylabel("observed frequency")
    plt.tight_layout()
    plt.savefig(out / "evaluation.png", dpi=140)
    plt.close(fig)

    imp = result.importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    xerr = np.vstack([imp["mean"] - imp["ci_low"], imp["ci_high"] - imp["mean"]]).clip(0)
    colors = ["#2a7" if s else "#aaa" for s in imp["consistent"]]
    ax.barh(imp["feature"], imp["mean"], xerr=xerr, color=colors)
    ax.axvline(0, c="grey", lw=1)
    ax.set_title("What the model leans on\n(green = positive in every split)")
    ax.set_xlabel("drop in PR-AUC when the feature is shuffled")
    plt.tight_layout()
    plt.savefig(out / "importance.png", dpi=140)
    plt.close(fig)


def write_report(result: TrainingResult, report_dir) -> Path:
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    ex = result.cfg["experiment"]

    pd.Series(result.funnel, name="employees").rename_axis("step").to_csv(out / "funnel.csv")
    br = result.by_rating.copy()
    small = br["n"] < MIN_CELL
    br.loc[small, ["success_simple_rate", "success_fair_rate"]] = np.nan
    br.loc[small, "n"] = np.nan
    br.round(3).to_csv(out / "success_by_rating.csv", index=False)

    result.grid.round(4).to_csv(out / "grid_results.csv", index=False)
    result.summary.round(4).to_csv(out / "grid_summary.csv", index=False)
    result.holdout.round(4).to_csv(out / "holdout.csv", index=False)
    result.calibration.round(4).to_csv(out / "calibration.csv", index=False)
    result.importance.round(4).to_csv(out / "importance.csv", index=False)
    result.significance.round(5).to_csv(out / "significance.csv", index=False)
    result.reduced["per_split"].round(4).to_csv(out / "reduced_model.csv", index=False)
    result.reduced["selection_frequency"].round(3).rename_axis("feature") \
        .to_csv(out / "reduced_selection.csv")
    result.subgroups.round(4).to_csv(out / "subgroups.csv", index=False)

    # 5th/95th percentiles instead of min/max: an extreme value can identify a person
    rows = []
    for c in result.numeric:
        s = result.X[c]
        rows.append({"feature": c, "type": "numeric", "non_missing": int(s.notna().sum()),
                     "mean": s.mean(), "std": s.std(), "p05": s.quantile(0.05),
                     "median": s.median(), "p95": s.quantile(0.95),
                     "top_value_share": s.value_counts(normalize=True).max()})
    for c in result.categorical:
        s = result.X[c]
        rows.append({"feature": c, "type": "categorical", "non_missing": int(s.notna().sum()),
                     "n_categories": int(s.nunique()),
                     "top_value_share": s.value_counts(normalize=True).max()})
    pd.DataFrame(rows).round(3).to_csv(out / "feature_summary.csv", index=False)

    meta = dict(result.artifact["metadata"])
    meta["reduced_model"] = {k: result.reduced[k] for k in
                             ("features", "mean_loss", "se_loss", "tolerance", "not_worse")}
    meta["selection_metric"] = ex["selection_metric"]
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=_jsonable)
    (out / "README.md").write_text(REPORT_README.format(min_cell=MIN_CELL,
                                                        min_sub=ex["min_subgroup_size"]))
    _plots(result, out)
    return out


def save_outputs(result: TrainingResult, outputs_dir) -> dict:
    out = Path(outputs_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = save_artifact(result.artifact, out / "rotation_fit_model.joblib")
    cohort_path = out / "cohort_features.xlsx"
    result.features.to_excel(cohort_path, index=False)
    return {"model": model_path, "cohort": cohort_path}
