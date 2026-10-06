"""Cohort construction and target labelling (notebook §2b)."""
from __future__ import annotations

import re

import pandas as pd

from .data import HRData, norm


def _contains_any(types: pd.Series, terms: list[str]) -> pd.Series:
    if not terms:
        return pd.Series(False, index=types.index)
    pattern = "|".join(re.escape(t.strip().casefold()) for t in terms)
    return types.str.contains(pattern, na=False, regex=True)


def valid_movements(hr: HRData, cfg: dict) -> pd.DataFrame:
    """Career rows that count as a genuine rotation."""
    c = cfg["cohort"]
    car = hr.career
    excl = [s.strip().casefold() for s in c["exclude_status"]]
    ok = ~norm(car["Employment Status"]).isin(excl).fillna(False)
    if c["check_old_status"]:
        ok &= ~norm(car["Old Employment Status"]).isin(excl).fillna(False)
    if c["keep_career_transition"]:
        keep = [f.strip().casefold() for f in c["keep_career_transition"]]
        ok &= norm(car["Career Transition"]).isin(keep).fillna(False)
    ok &= ~_contains_any(norm(car["Transition Type"]), c["exclude_types"])
    ok &= car["_k"].isin(set(hr.emp["_k"]))
    return car[ok.astype(bool)].copy()


def rating_ordinal(s: pd.Series, cfg: dict) -> pd.Series:
    order = {k.strip().casefold(): v for k, v in cfg["ratings"]["order"].items()}
    return norm(s).map(order).astype("Float64").astype(float)


def build_cohort(hr: HRData, cfg: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Returns (cohort, inclusion funnel, success rate by pre-rotation rating)."""
    c, r = cfg["cohort"], cfg["ratings"]
    emp = hr.emp
    funnel = {"Employees in dataset": int(emp["_k"].nunique())}

    mv = valid_movements(hr, cfg)
    funnel["Rotated at any time (valid movement)"] = int(mv["_k"].nunique())

    win = mv[mv["Start Date"].between(pd.Timestamp(c["window_start"]),
                                      pd.Timestamp(c["window_end"]))].copy()
    funnel[f"Rotated {c['window_start']} to {c['window_end']}"] = int(win["_k"].nunique())

    # Anchor = LAST qualifying move inside the window. Later moves (the Sept 2025
    # restructure) are contamination of the outcome, not the studied event.
    anchor = win.sort_values("Start Date", kind="stable").groupby("_k").tail(1).copy()
    anchor["position_changed"] = (norm(anchor["Old Position"])
                                  != norm(anchor["Position"])).fillna(True)
    anchor["org_unit_changed"] = (norm(anchor["Old Organization Unit"])
                                  != norm(anchor["Organization Unit"])).fillna(True)
    rule = str(c.get("real_change_rule") or "none").lower()
    if rule == "or":
        anchor = anchor[anchor["position_changed"] | anchor["org_unit_changed"]]
    elif rule == "and":
        anchor = anchor[anchor["position_changed"] & anchor["org_unit_changed"]]
    funnel["Position or org unit actually changed"] = int(anchor["_k"].nunique())

    anchor = anchor[["_k", "Start Date", "Transition Type", "Old Position", "Position",
                     "Old Organization Unit", "Organization Unit", "Old Grade", "Grade",
                     "position_changed", "org_unit_changed"]].rename(columns={
        "Start Date": "rotation_date",
        "Old Position": "origin_position", "Position": "destination_position",
        "Old Organization Unit": "origin_org_unit",
        "Organization Unit": "destination_org_unit",
        "Old Grade": "origin_grade", "Grade": "destination_grade",
    })

    o_before = rating_ordinal(emp[r["before"]], cfg)
    o_after = rating_ordinal(emp[r["after"]], cfg)
    ratings = pd.DataFrame({"_k": emp["_k"], "rating_before_ord": o_before,
                            "rating_after_ord": o_after})

    df = anchor.merge(ratings, on="_k", how="inner")
    df = df[df["rating_before_ord"].notna() & df["rating_after_ord"].notna()].copy()
    funnel["Both ratings present"] = int(len(df))

    # --- label 1: naive ---
    df["success_simple"] = (df["rating_after_ord"] >= df["rating_before_ord"]).astype(int)

    # --- label 2: fair (recommended) ---
    # Baseline = mean post-rotation rating among everyone who started at the
    # same pre-rotation level. Success = beat that baseline. Removes the
    # company-wide recalibration and does not penalise starting high.
    if c["baseline_population"] == "all":
        both = ratings.dropna(subset=["rating_before_ord", "rating_after_ord"])
        expected = both.groupby("rating_before_ord")["rating_after_ord"].mean()
    else:
        expected = df.groupby("rating_before_ord")["rating_after_ord"].mean()
    df["expected_after"] = df["rating_before_ord"].map(expected)
    df["residual_after"] = df["rating_after_ord"] - df["expected_after"]
    df["success_fair"] = (df["residual_after"] > 0).astype(int)

    cohort = df.reset_index(drop=True)
    by_rating = (cohort.groupby("rating_before_ord")
                 .agg(n=("success_fair", "size"),
                      success_simple_rate=("success_simple", "mean"),
                      success_fair_rate=("success_fair", "mean"))
                 .reset_index())
    return cohort, funnel, by_rating
