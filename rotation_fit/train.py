"""Training, model selection, interpretation and the final model (notebook §5-§9).

Changes from the notebook, and why:
* A stratified HOLD-OUT set is split off first and touched once, at the end.
  The split-ratio x seed x scaler grid runs on the remaining development set,
  so choosing the best model no longer leaks into its reported score.
* The deployed model is refitted on ALL labelled rows (not 70 %), and
  calibrated with sigmoid scaling (isotonic overfits ~150 positives).
* Permutation importance permutes RAW columns through the whole pipeline and
  is aggregated over every split, giving an interval instead of one number.
* Significance is tested properly (drop-one likelihood-ratio tests with Holm
  correction, development rows only). A "significant attributes only" model
  is evaluated with that selection redone INSIDE each split, so the
  comparison is not circular.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors

from .cohort import build_cohort
from .data import HRData, load_hr_data
from .features import CATEGORICAL_CANDIDATES, NUMERIC_CANDIDATES, FeatureBuilder
from .models import make_pipeline, make_preprocessor, prepare_X, resolve_ann_backend
from .significance import lr_tests, select_significant

ARTIFACT_VERSION = 2
LOWER_IS_BETTER = {"brier"}


def metrics(y, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    two = len(np.unique(y)) > 1
    return {"auc": roc_auc_score(y, proba) if two else np.nan,
            "pr_auc": average_precision_score(y, proba) if two else np.nan,
            "f1": f1_score(y, pred, zero_division=0),
            "brier": brier_score_loss(y, proba)}


def _ratio(test_size: float) -> str:
    return f"{round((1 - test_size) * 100)}:{round(test_size * 100)}"


class OODDetector:
    """Flags pairings unlike anything in training: a confident number for an
    unseen kind of case is worse than no number at all."""

    def __init__(self, numeric, categorical, k=5, quantile=95):
        self.numeric, self.categorical, self.k, self.quantile = numeric, categorical, k, quantile

    def fit(self, X: pd.DataFrame):
        self.pre_ = make_preprocessor("standard", self.numeric, self.categorical).fit(X)
        Z = self.pre_.transform(X)
        k = min(self.k + 1, len(Z))
        self.nn_ = NearestNeighbors(n_neighbors=k).fit(Z)
        # skip column 0: every reference row is its own nearest neighbour
        d = self.nn_.kneighbors(Z)[0][:, 1:].mean(axis=1)
        self.threshold_ = float(np.percentile(d, self.quantile))
        return self

    def distance(self, X: pd.DataFrame) -> np.ndarray:
        d = self.nn_.kneighbors(self.pre_.transform(X), n_neighbors=self.nn_.n_neighbors - 1)[0]
        return d.mean(axis=1)

    def flag(self, X: pd.DataFrame) -> np.ndarray:
        return self.distance(X) > self.threshold_


@dataclass
class TrainingResult:
    cfg: dict
    funnel: dict
    by_rating: pd.DataFrame
    features: pd.DataFrame            # engineered table — PERSONAL DATA
    X: pd.DataFrame
    y: np.ndarray
    numeric: list
    categorical: list
    grid: pd.DataFrame
    summary: pd.DataFrame
    best: dict
    holdout: pd.DataFrame
    calibration: pd.DataFrame
    importance: pd.DataFrame
    significance: pd.DataFrame
    reduced: dict
    subgroups: pd.DataFrame
    artifact: dict
    notes: list = field(default_factory=list)


def _log(progress, msg):
    if progress:
        progress(msg)


def _choose_features(feats: pd.DataFrame, notes: list):
    num = [c for c in NUMERIC_CANDIDATES if c in feats and feats[c].notna().any()]
    cat = [c for c in CATEGORICAL_CANDIDATES if c in feats and feats[c].notna().any()]
    for c in num + cat:
        top = feats[c].value_counts(normalize=True, dropna=False).iloc[0]
        if top > 0.95:
            notes.append(f"{c}: one value covers {top:.0%} of the cohort (near constant).")
    return num, cat


def _grid(X, y, dev, cfg, num, cat, backend, progress):
    ex = cfg["experiment"]
    rows, splits = [], []
    for test_size in ex["split_ratios"]:
        for seed in ex["seeds"]:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            tr, te = next(sss.split(dev, y[dev]))
            splits.append((test_size, seed, dev[tr], dev[te]))
    for scaler in ex["scalers"]:
        for test_size, seed, tr, te in splits:
            for name in ex["models"]:
                t0 = time.time()
                pipe = make_pipeline(name, scaler, seed, num, cat, backend)
                pipe.fit(X.iloc[tr], y[tr])
                proba = pipe.predict_proba(X.iloc[te])[:, 1]
                rows.append({"model": name, "scaler": scaler, "ratio": _ratio(test_size),
                             "seed": seed, **metrics(y[te], proba),
                             "fit_time": time.time() - t0})
        _log(progress, f"grid: finished scaler '{scaler}'")
    return pd.DataFrame(rows), splits


def _importance(X, y, splits, best, cfg, num, cat, backend, cols=None):
    """Permutation importance of raw columns, one value per split."""
    cols = cols or (num + cat)
    per_split = []
    for test_size, seed, tr, te in splits:
        pipe = make_pipeline(best["model"], best["scaler"], seed,
                             [c for c in num if c in cols], [c for c in cat if c in cols], backend)
        pipe.fit(X.iloc[tr][cols], y[tr])
        pi = permutation_importance(pipe, X.iloc[te][cols], y[te],
                                    n_repeats=cfg["experiment"]["importance_repeats"],
                                    random_state=seed, scoring="average_precision")
        per_split.append(pd.Series(pi.importances_mean, index=cols))
    return pd.DataFrame(per_split)


def _summarise_importance(per_split: pd.DataFrame) -> pd.DataFrame:
    imp = pd.DataFrame({
        "mean": per_split.mean(), "std": per_split.std(ddof=1),
        "ci_low": per_split.quantile(0.025), "ci_high": per_split.quantile(0.975),
        "share_of_splits_positive": (per_split > 0).mean(),
    })
    # Splits share most rows, so this interval is optimistic: "consistent" means
    # the model leans on it every time, NOT that it is statistically significant
    # (see significance.csv for that).
    imp["consistent"] = imp["ci_low"] > 0
    return imp.sort_values("mean", ascending=False).rename_axis("feature").reset_index()


def _select(X, y, rows, cfg, best, num, cat, backend, seed):
    """Pick the reduced model's attributes using ONLY the given training rows."""
    ex = cfg["experiment"]
    k = ex["reduced_top_k"]
    if ex.get("reduced_selection", "lr_test") == "lr_test":
        return select_significant(X.iloc[rows], y[rows], num, cat,
                                  ex.get("reduced_alpha", 0.05), k)
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    itr, iva = next(inner.split(rows, y[rows]))
    per = _importance(X, y, [(0.25, seed, rows[itr], rows[iva])], best, cfg, num, cat, backend)
    return list(per.iloc[0].sort_values(ascending=False).index[:k])


def _reduced_comparison(X, y, splits, best, cfg, num, cat, backend, grid):
    """Significant-attributes model, with selection done inside each split
    (no double dipping), compared with the full model on the same splits."""
    metric = cfg["experiment"]["selection_metric"]
    rows, picked = [], []
    for test_size, seed, tr, te in splits:
        top = _select(X, y, tr, cfg, best, num, cat, backend, seed)
        picked.extend(top)
        pipe = make_pipeline(best["model"], best["scaler"], seed,
                             [c for c in num if c in top], [c for c in cat if c in top], backend)
        pipe.fit(X.iloc[tr][top], y[tr])
        m = metrics(y[te], pipe.predict_proba(X.iloc[te][top])[:, 1])
        full = grid[(grid.model == best["model"]) & (grid.scaler == best["scaler"])
                    & (grid.ratio == _ratio(test_size)) & (grid.seed == seed)].iloc[0]
        rows.append({"ratio": _ratio(test_size), "seed": seed, "n_features": len(top),
                     f"full_{metric}": full[metric], f"reduced_{metric}": m[metric]})
    per_split = pd.DataFrame(rows)
    diff = per_split[f"full_{metric}"] - per_split[f"reduced_{metric}"]
    if metric in LOWER_IS_BETTER:
        diff = -diff
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
    tol = cfg["experiment"].get("reduced_tolerance", 0.01)
    return {
        "per_split": per_split,
        "selection_frequency": (pd.Series(picked).value_counts() / len(splits)).rename("share"),
        "mean_loss": float(diff.mean()),          # how much worse the reduced model is
        "se_loss": float(se),
        "tolerance": tol,
        # simpler model accepted only if it costs at most `tol` of the metric
        "not_worse": bool(diff.mean() <= tol),
    }


def _fit_final(X, y, best, cfg, num, cat, backend):
    pipe = make_pipeline(best["model"], best["scaler"], cfg["experiment"]["holdout_seed"],
                         num, cat, backend)
    method = cfg["experiment"]["calibration"]
    if method in ("sigmoid", "isotonic") and best["model"] != "dummy":
        model = CalibratedClassifierCV(pipe, method=method, cv=5)
    else:
        model = pipe
    return model.fit(X, y)


def train(cfg: dict, hr: HRData | None = None, progress=None) -> TrainingResult:
    ex = cfg["experiment"]
    hr = hr or load_hr_data(cfg)
    notes = list(hr.notes)

    _log(progress, "building cohort")
    cohort, funnel, by_rating = build_cohort(hr, cfg)
    target = cfg["cohort"]["target"]
    y = cohort[target].astype(int).to_numpy()
    if len(cohort) < 30 or y.sum() < 10 or (len(y) - y.sum()) < 10:
        raise ValueError(f"cohort too small to train: {len(cohort)} rows, "
                         f"{int(y.sum())} positives")

    _log(progress, "engineering features")
    fb = FeatureBuilder(cfg)
    feats = fb.fit_transform(hr, cohort, cfg["ratings"]["before"])
    feats[target] = y
    num, cat = _choose_features(feats, notes)
    X = prepare_X(feats, num, cat)
    backend = resolve_ann_backend(ex["ann_backend"])

    # ---- hold-out set: touched once, at the very end -------------------
    sss = StratifiedShuffleSplit(n_splits=1, test_size=ex["holdout_size"],
                                 random_state=ex["holdout_seed"])
    dev, test = next(sss.split(X, y))

    _log(progress, f"running grid on {len(dev)} development rows")
    grid, splits = _grid(X, y, dev, cfg, num, cat, backend, progress)

    metric = ex["selection_metric"]
    summary = (grid.groupby(["model", "scaler"])
               .agg(auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                    pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                    f1_mean=("f1", "mean"), brier_mean=("brier", "mean"),
                    fit_time=("fit_time", "mean"))
               .reset_index())
    cand = summary[summary.model != "dummy"]
    cand = cand.sort_values(f"{metric}_mean", ascending=metric in LOWER_IS_BETTER)
    best = cand.iloc[0][["model", "scaler"]].to_dict()
    best["dev_" + metric] = float(cand.iloc[0][f"{metric}_mean"])
    _log(progress, f"best on development set: {best['model']} / {best['scaler']}")

    # ---- interpretation -------------------------------------------------
    _log(progress, "permutation importance across splits")
    best_splits = [s for s in splits if s[0] == ex["split_ratios"][0]] or splits
    importance = _summarise_importance(
        _importance(X, y, best_splits, best, cfg, num, cat, backend))

    _log(progress, "significance tests (development rows only)")
    significance = lr_tests(X.iloc[dev], y[dev], num, cat)

    _log(progress, "reduced-model comparison")
    reduced = _reduced_comparison(X, y, best_splits, best, cfg, num, cat, backend, grid)
    reduced["features"] = _select(X, y, dev, cfg, best, num, cat, backend, ex["holdout_seed"])
    k = len(reduced["features"])

    # ---- hold-out evaluation (once) -------------------------------------
    _log(progress, "evaluating on the hold-out set")
    hold_rows, proba_best = [], None
    for name in ex["models"]:
        sc = summary[summary.model == name].sort_values(
            f"{metric}_mean", ascending=metric in LOWER_IS_BETTER).iloc[0]["scaler"]
        pipe = make_pipeline(name, sc, ex["holdout_seed"], num, cat, backend)
        pipe.fit(X.iloc[dev], y[dev])
        hold_rows.append({"model": name, "scaler": sc, "variant": "uncalibrated",
                          **metrics(y[test], pipe.predict_proba(X.iloc[test])[:, 1])})
    final_dev = _fit_final(X.iloc[dev], y[dev], best, cfg, num, cat, backend)
    proba_best = final_dev.predict_proba(X.iloc[test])[:, 1]
    hold_rows.append({"model": best["model"], "scaler": best["scaler"],
                      "variant": f"calibrated ({ex['calibration']})",
                      **metrics(y[test], proba_best)})
    rnum, rcat = [c for c in num if c in reduced["features"]], [c for c in cat if c in reduced["features"]]
    red_dev = _fit_final(X.iloc[dev][rnum + rcat], y[dev], best, cfg, rnum, rcat, backend)
    hold_rows.append({"model": best["model"], "scaler": best["scaler"],
                      "variant": f"calibrated, {k} significant features",
                      **metrics(y[test], red_dev.predict_proba(X.iloc[test][rnum + rcat])[:, 1])})
    holdout = pd.DataFrame(hold_rows)

    n_bins = int(np.clip(len(test) // 15, 3, 8))
    frac, mean_pred = calibration_curve(y[test], proba_best, n_bins=n_bins, strategy="quantile")
    calibration = pd.DataFrame({"mean_predicted": mean_pred, "observed_rate": frac})

    # ---- subgroup check on the hold-out set ------------------------------
    sub_rows = []
    sub = feats.iloc[test].assign(_p=proba_best, _y=y[test])
    for col in [c for c in ("Level Jabatan", "gender") if c in sub.columns]:
        for g, grp in sub.groupby(col):
            if grp["_y"].nunique() > 1 and len(grp) >= ex["min_subgroup_size"]:
                sub_rows.append({"attribute": col, "group": str(g), "n": len(grp),
                                 "auc": roc_auc_score(grp["_y"], grp["_p"])})
    subgroups = pd.DataFrame(sub_rows, columns=["attribute", "group", "n", "auc"])

    # ---- final model: ALL labelled rows ---------------------------------
    deploy = ex.get("deploy", "auto")
    use_reduced = deploy == "reduced" or (deploy == "auto" and reduced["not_worse"])
    fnum, fcat = (rnum, rcat) if use_reduced else (num, cat)
    _log(progress, f"fitting final model on all {len(X)} rows "
                   f"({str(k) + ' significant' if use_reduced else 'all'} features)")
    final = _fit_final(X[fnum + fcat], y, best, cfg, fnum, fcat, backend)
    ood = OODDetector(fnum, fcat).fit(X[fnum + fcat])
    baseline = {c: (float(X[c].median()) if c in fnum else X[c].mode(dropna=True).iloc[0]
                    if X[c].notna().any() else np.nan) for c in fnum + fcat}

    hold_best = holdout.iloc[-2 if not use_reduced else -1]
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "target": target,
        "cohort_size": int(len(cohort)),
        "positive_rate": float(y.mean()),
        "holdout_size": int(len(test)),
        "window": [cfg["cohort"]["window_start"], cfg["cohort"]["window_end"]],
        "best": {"model": best["model"], "scaler": best["scaler"],
                 "calibration": ex["calibration"], "ann_backend": backend},
        "deployed_features": fnum + fcat,
        "deployed_variant": "reduced" if use_reduced else "full",
        "holdout_metrics": {m: float(hold_best[m]) for m in ("auc", "pr_auc", "f1", "brier")},
        "baseline_pr_auc": float(y[test].mean()),
        "rating_used_for_scoring": cfg["ratings"]["current"],
        "pending_inputs": {"job_family_map": hr.jf_map is not None,
                           "hris_demographics": "gender" in hr.emp.columns},
        "known_limitations": [
            "2024 was the company's first rating cycle and was lenient; 2025 tightened.",
            "The Sept 2025 restructure affects the final quarter of the 2025 rating.",
            "Group membership for the similarity fallback uses the current structure.",
            "Probabilities describe rotations like those in the training window; "
            "check the out_of_distribution flag before relying on a number.",
        ],
        "notes": notes,
        "library_versions": {"pandas": pd.__version__, "numpy": np.__version__,
                             "sklearn": sklearn.__version__},
    }
    artifact = {"version": ARTIFACT_VERSION, "feature_builder": fb, "model": final,
                "numeric": fnum, "categorical": fcat, "features": fnum + fcat,
                "ood": ood, "baseline": baseline, "importance": importance,
                "significance": significance,
                "metadata": metadata, "cfg": cfg}
    _log(progress, "done")
    return TrainingResult(cfg=cfg, funnel=funnel, by_rating=by_rating, features=feats, X=X, y=y,
                          numeric=num, categorical=cat, grid=grid, summary=summary, best=best,
                          holdout=holdout, calibration=calibration, importance=importance,
                          significance=significance,
                          reduced=reduced, subgroups=subgroups, artifact=artifact, notes=notes)
