import json
import pickle

import numpy as np
import pandas as pd

from rotation_fit import RotationScorer, load_artifact
from rotation_fit.report import save_outputs, write_report


def test_training_result_shape(result):
    assert result.best["model"] != "dummy"
    assert {"uncalibrated"} <= set(result.holdout["variant"].str.split(" ").str[0])
    assert len(result.importance) == len(result.numeric) + len(result.categorical)
    assert result.artifact["metadata"]["cohort_size"] == len(result.y)
    # the synthetic data carries a real signal through Job Fit
    best = result.holdout[result.holdout.variant.str.startswith("calibrated (")].iloc[0]
    assert best["auc"] > 0.6


def test_holdout_is_separate_from_grid(result):
    n_dev = len(result.y) - result.artifact["metadata"]["holdout_size"]
    assert n_dev < len(result.y)


def test_artifact_has_no_hr_snapshot(result):
    fb_state = result.artifact["feature_builder"].__getstate__()
    assert "attrs_" not in fb_state and "_career_by_k" not in fb_state
    blob = pickle.dumps(result.artifact)
    assert b"Employee 0001" not in blob     # no names inside the model file


def test_save_load_score_rank_explain(result, hr, cfg, tmp_path):
    paths = save_outputs(result, tmp_path)
    scorer = RotationScorer(load_artifact(paths["model"]), hr)
    unit = scorer.destination_units()[0]
    emp = hr.emp.head(3)
    pairs = pd.DataFrame({"NIK": emp["NIK"], "destination_position": "Tax Analyst",
                          "destination_org_unit": unit})
    out = scorer.score_pairs(pairs)
    assert out["success_pct"].between(0, 100).all()
    assert out["out_of_distribution"].dtype == bool

    ranked = scorer.rank_candidates("Tax Analyst", unit)
    assert ranked["success_pct"].is_monotonic_decreasing
    assert not (ranked["origin_position"] == "Tax Analyst").any()

    ex = scorer.explain(pairs.head(1))
    assert set(ex["feature"]) == set(result.artifact["features"])


def test_report_contains_no_personal_data(result, hr, tmp_path):
    out = write_report(result, tmp_path / "report")
    text = "".join(p.read_text(errors="ignore") for p in out.iterdir()
                   if p.suffix in {".csv", ".json", ".md"})
    for nik in hr.emp["_k"].head(200):
        assert nik not in text
    for name in hr.emp["Nama"].head(200):
        assert name not in text
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["target"] == "success_fair"
    assert (out / "evaluation.png").exists() and (out / "importance.png").exists()


def test_reduced_model_comparison(result, cfg):
    r = result.reduced
    assert 1 <= len(r["features"]) <= cfg["experiment"]["reduced_top_k"]
    assert np.isfinite(r["mean_loss"])


def test_significance_finds_the_planted_signal(result):
    """Synthetic success depends on Job Fit; marital status is pure noise."""
    sig = result.significance.set_index("feature")
    assert sig.loc["eqs_kompetensi", "p_holm"] < 0.05
    assert sig.loc["eqs_kompetensi", "direction"] == "+"
    assert sig.loc["marital", "p_holm"] > 0.05
    assert (sig["p_holm"] >= sig["p_value"] - 1e-12).all()
