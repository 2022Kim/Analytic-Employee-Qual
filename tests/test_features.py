import numpy as np
import pandas as pd
import pytest

from rotation_fit.cohort import build_cohort, valid_movements
from rotation_fit.data import key
from rotation_fit.features import (FeatureBuilder, PercentileScaler, aspirasi_score,
                                   edu_years, gower_similarity, rekam_jejak)

POINTS = {"Aspirasi Individual": 20, "Aspirasi Jobholder": 25,
          "Aspirasi Unit": 25, "Aspirasi Supervisor": 30}


def test_key_keeps_text_ids_and_fixes_excel_floats():
    s = pd.Series(["1486.020-R", 1234.0, " 99 "])
    assert list(key(s)) == ["1486.020-R", "1234", "99"]


@pytest.mark.parametrize("raw,years", [("S1 Teknik", 16), ("D3", 15), ("SMA", 12),
                                       ("S2 Manajemen", 18), ("D3 / S1", 16),
                                       ("unknown", np.nan)])
def test_edu_years(raw, years):
    assert edu_years(raw) == years or (np.isnan(years) and np.isnan(edu_years(raw)))


def test_rekam_jejak_blank_means_clean():
    aktif = pd.Series(["SP1", None, None, "  "])
    past = pd.Series([None, "SP2 2020", None, None])
    assert list(rekam_jejak(aktif, past)) == [0, 50, 100, 100]


def test_aspirasi_takes_highest_matching_column():
    row = pd.Series({"Aspirasi Individual": "- Tax Analyst",
                     "Aspirasi Supervisor": "- Senior Tax Analyst\n- Other",
                     "Aspirasi Unit": None, "Aspirasi Jobholder": "Payroll"})
    assert aspirasi_score(row, "Tax Analyst", POINTS, 0.35) == 30
    assert aspirasi_score(row, "Network Engineer", POINTS, 0.35) == 0
    assert aspirasi_score(row, "", POINTS, 0.35) == 0


def test_percentile_scaler_range_and_order():
    sc = PercentileScaler().fit([0.1, 0.2, 0.3, 0.4, np.nan])
    out = sc.transform([0.0, 0.25, 1.0, np.nan])
    assert out[0] == 1.0 and out[2] == 100.0 and out[0] < out[1] < out[2]
    assert np.isnan(out[3])


def test_gower_identical_is_one_and_ignores_missing():
    a_num, B_num = np.array([[30.0, 16.0]]), np.array([[30.0, 16.0], [30.0, np.nan]])
    a_cat = np.array([["M"]], dtype=object)
    B_cat = np.array([["M"], [None]], dtype=object)
    assert gower_similarity(a_num, B_num, np.array([10.0, 5.0]), a_cat, B_cat) == 1.0


def test_cohort_rules(hr, cfg):
    cohort, funnel, _ = build_cohort(hr, cfg)
    steps = list(funnel.values())
    assert all(a >= b for a, b in zip(steps[1:], steps[2:])), "funnel must not grow"
    assert cohort["_k"].is_unique
    assert set(cohort["success_fair"].unique()) <= {0, 1}
    mv = valid_movements(hr, cfg)
    assert not mv["Transition Type"].str.contains("Promotion|Pelaksana Tugas").any()
    assert not (mv["Employment Status"] == "Outsource").any()
    start, end = pd.Timestamp(cfg["cohort"]["window_start"]), pd.Timestamp(cfg["cohort"]["window_end"])
    assert cohort["rotation_date"].between(start, end).all()


def test_similarity_uses_demographics(hr, cfg):
    """Regression: the notebook merged gender/marital into the cohort only, so
    similarity silently ignored them."""
    cohort, _, _ = build_cohort(hr, cfg)
    fb = FeatureBuilder(cfg)
    fb.fit_transform(hr, cohort.head(50), cfg["ratings"]["before"])
    _, cat = fb._sim_dims(hr)
    assert {"gender", "marital", "job_family"} <= set(cat)


def test_feature_builder_matches_between_train_and_score(hr, cfg):
    cohort, _, _ = build_cohort(hr, cfg)
    fb = FeatureBuilder(cfg)
    train_f = fb.fit_transform(hr, cohort, cfg["ratings"]["before"])
    again = fb.transform(hr, cohort, cfg["ratings"]["before"])
    cols = ["eqs_kinerja", "eqs_pelatihan", "eqs_aspirasi", "similarity_score", "usia"]
    pd.testing.assert_frame_equal(train_f[cols], again[cols])
    assert train_f["similarity_score"].dropna().between(1, 100).all()


def test_unknown_employee_is_rejected(hr, cfg):
    cohort, _, _ = build_cohort(hr, cfg)
    fb = FeatureBuilder(cfg).fit(hr, cohort.head(40), cfg["ratings"]["before"])
    bad = pd.DataFrame({"_k": ["does-not-exist"], "rotation_date": ["2026-01-01"],
                        "destination_position": ["x"], "destination_org_unit": ["y"]})
    with pytest.raises(ValueError, match="not found"):
        fb.transform(hr, bad, cfg["ratings"]["current"])
