"""Rotation Fit — internal webapp.

Run from the repo root:   streamlit run app/streamlit_app.py

Data stays where it is: the app reads the Excel exports (from the paths in
config.yaml, or uploaded in the sidebar) and keeps them in memory only.
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rotation_fit import RotationScorer, load_artifact, load_config, load_hr_data, train  # noqa: E402
from rotation_fit.predict import PAIR_COLUMNS  # noqa: E402
from rotation_fit.report import save_outputs, write_report  # noqa: E402
from rotation_fit.synthetic import make_synthetic, synthetic_config_overrides  # noqa: E402

st.set_page_config(page_title="Rotation Fit", page_icon="🔄", layout="wide")

DEMO_EXPERIMENT = {"split_ratios": [0.3], "seeds": [1, 2, 3], "scalers": ["standard"],
                   "models": ["dummy", "logreg", "rf"], "importance_repeats": 3}


# ------------------------------------------------------------------ loading
def _named_buffer(upload):
    if upload is None:
        return None
    buf = io.BytesIO(upload.getvalue())
    buf.name = upload.name
    return buf


@st.cache_resource(show_spinner="Loading HR data…")
def _load_hr(cfg_key: str, _cfg: dict, _career=None, _dataset=None, _jf=None, _dg=None):
    # cfg_key identifies the files; the underscored arguments are not hashed
    return load_hr_data(_cfg, career=_career, dataset=_dataset, jobfamily_map=_jf,
                        demographics=_dg)


@st.cache_resource(show_spinner="Creating synthetic demo data…")
def _demo_paths():
    return make_synthetic(Path(tempfile.mkdtemp(prefix="rotation_fit_demo_")), n_employees=700)


@st.cache_resource(show_spinner="Loading model…")
def _load_model(path: str, mtime: float):
    return load_artifact(path)


def _hbar(df: pd.DataFrame, value: str, label: str = "feature", title: str = ""):
    """Sorted horizontal bars; green helps, red hurts."""
    import altair as alt

    d = df[[label, value]].copy()
    d["sign"] = (d[value] >= 0).map({True: "positive", False: "negative"})
    chart = alt.Chart(d).mark_bar().encode(
        x=alt.X(f"{value}:Q", title=title),
        y=alt.Y(f"{label}:N", sort="-x", title=None),
        color=alt.Color("sign:N", legend=None,
                        scale=alt.Scale(domain=["positive", "negative"],
                                        range=["#2e9e6a", "#d1495b"])),
        tooltip=[label, alt.Tooltip(f"{value}:Q", format=".3f")],
    ).properties(height=max(120, 26 * len(d)))
    st.altair_chart(chart, width="stretch")


def _excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🔄 Rotation Fit")
source = st.sidebar.radio("Data source", ["Files in config.yaml", "Upload spreadsheets",
                                          "Synthetic demo data"])

cfg = load_config()
hr, cfg_key = None, source
try:
    if source == "Files in config.yaml":
        hr = _load_hr("config", cfg)
    elif source == "Upload spreadsheets":
        car_u = st.sidebar.file_uploader("Career transactions (.xlsx/.csv)", ["xlsx", "xls", "csv"])
        emp_u = st.sidebar.file_uploader("Dataset (.xlsx/.csv)", ["xlsx", "xls", "csv"])
        jf_u = st.sidebar.file_uploader("Job family map (optional)", ["xlsx", "xls", "csv"])
        dg_u = st.sidebar.file_uploader("HRIS demographics (optional)", ["xlsx", "xls", "csv"])
        if car_u and emp_u:
            cfg_key = "|".join(f"{u.name}:{u.size}" for u in (car_u, emp_u, jf_u, dg_u) if u)
            hr = _load_hr(cfg_key, cfg, _named_buffer(car_u), _named_buffer(emp_u),
                          _named_buffer(jf_u), _named_buffer(dg_u))
        else:
            st.sidebar.info("Upload at least the career file and the dataset.")
    else:
        demo = _demo_paths()
        cfg = load_config(overrides={**synthetic_config_overrides(demo),
                                     "experiment": DEMO_EXPERIMENT})
        cfg["paths"]["outputs"] = str(Path(demo["dataset"]).parent / "outputs")
        cfg["paths"]["report"] = str(Path(demo["dataset"]).parent / "report")
        hr = _load_hr("demo", cfg)
        st.sidebar.caption("Randomly generated employees — for trying the app only.")
except Exception as e:  # noqa: BLE001 — show any loading problem to the user
    st.sidebar.error(f"Could not load data: {e}")

model_path = Path(cfg["paths"]["outputs"]) / "rotation_fit_model.joblib"
art = None
if model_path.exists():
    try:
        art = _load_model(str(model_path), model_path.stat().st_mtime)
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"Model file could not be loaded: {e}")
st.sidebar.markdown("---")
if art:
    m = art["metadata"]
    st.sidebar.success(f"Model: **{m['best']['model']}** ({m['deployed_variant']} features)\n\n"
                       f"trained {m['trained_at'][:10]} · hold-out AUC "
                       f"{m['holdout_metrics']['auc']:.2f}")
else:
    st.sidebar.warning("No trained model yet — use the **Model** tab to train one.")
if hr is not None:
    st.sidebar.caption(f"{len(hr.emp):,} employees · {len(hr.career):,} career rows")
    for n in hr.notes:
        st.sidebar.caption(f"ℹ️ {n}")

st.info("**Decision support, not a decision.** The percentage estimates how often rotations "
        "*like this one* succeeded in the past. Always combine it with the manager's and HR's "
        "judgement, and treat ⚠️ unusual cases with extra care.", icon="🧭")

scorer = RotationScorer(art, hr) if (art and hr is not None) else None
tab_one, tab_rank, tab_batch, tab_model = st.tabs(
    ["Score a rotation", "Rank candidates for a role", "Batch (spreadsheet)", "Model"])


def _need_scorer():
    if scorer is None:
        st.warning("Load data and train (or provide) a model first.")
        return False
    return True


def _role_inputs(prefix: str):
    positions = scorer.destination_positions()
    units = scorer.destination_units()
    c1, c2, c3 = st.columns([2, 2, 1])
    pos = c1.selectbox("Destination position", positions, key=f"{prefix}_pos")
    unit = c2.selectbox("Destination organisation unit", units, key=f"{prefix}_unit")
    date = c3.date_input("Rotation date", pd.Timestamp.today(), key=f"{prefix}_date")
    return pos, unit, date


# ------------------------------------------------------------------ tab 1
with tab_one:
    if _need_scorer():
        emp = hr.emp
        labels = (emp["_k"] + " — " + emp["Nama"].astype("string").fillna("")).tolist()
        choice = st.selectbox("Employee", labels)
        pos, unit, date = _role_inputs("one")
        if st.button("Estimate", type="primary"):
            nik = choice.split(" — ")[0]
            pair = pd.DataFrame({"_k": [nik], "destination_position": [pos],
                                 "destination_org_unit": [unit], "rotation_date": [date]})
            res = scorer.score_pairs(pair).iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Estimated success probability", f"{res['success_pct']:.0f}%")
            c2.metric("Similarity measured against",
                      {"destination_org_unit": "Destination team", "Group": "Destination group",
                       "insufficient": "—"}.get(res["similarity_basis"], res["similarity_basis"]))
            c3.metric("Current position", str(res["origin_position"]))
            if res["out_of_distribution"]:
                st.warning("⚠️ Unusual case: this employee/role combination is unlike the "
                           "rotations the model learned from. Treat the number with caution.")
            if res["similarity_basis"] == "insufficient":
                st.warning("The destination team is too small to compute similarity.")
            st.subheader("What moves this estimate")
            ex = scorer.explain(pair)
            st.caption("Percentage points gained (+) or lost (−) compared with a typical "
                       "employee value for each attribute.")
            _hbar(ex, "effect_pct_points", title="percentage points")
            show = ex.copy()
            for c in ("value", "typical_value"):
                show[c] = [f"{v:.1f}" if isinstance(v, float) else str(v) for v in show[c]]
            st.dataframe(show, hide_index=True, width="stretch")

# ------------------------------------------------------------------ tab 2
with tab_rank:
    if _need_scorer():
        pos, unit, date = _role_inputs("rank")
        c1, c2 = st.columns(2)
        div = c1.multiselect("Limit candidates to Divisi (optional)",
                             sorted(hr.emp["Divisi"].dropna().astype(str).unique()))
        top = c2.number_input("Show top", 5, 500, 25)
        hide_ood = st.checkbox("Hide unusual cases (⚠️)", value=False)
        if st.button("Rank", type="primary"):
            pool = hr.emp if not div else hr.emp[hr.emp["Divisi"].astype(str).isin(div)]
            with st.spinner(f"Scoring {len(pool):,} candidates…"):
                ranked = scorer.rank_candidates(pos, unit, candidates=pool["_k"], date=date)
            if hide_ood:
                ranked = ranked[~ranked["out_of_distribution"]]
            st.dataframe(ranked.head(int(top)), hide_index=True, width="stretch")
            st.download_button("Download full ranking (.xlsx)", _excel_bytes(ranked),
                               file_name="rotation_ranking.xlsx")

# ------------------------------------------------------------------ tab 3
with tab_batch:
    st.markdown("Upload a spreadsheet with one row per proposed rotation. Required columns: "
                f"`{'`, `'.join(PAIR_COLUMNS)}`; optional: `rotation_date`, `destination_group`.")
    st.download_button("Download template", pd.DataFrame(columns=PAIR_COLUMNS + ["rotation_date"])
                       .to_csv(index=False).encode(), file_name="score_pairs_template.csv")
    up = st.file_uploader("Pairs to score", ["xlsx", "xls", "csv"], key="pairs")
    if up and _need_scorer():
        from rotation_fit.data import read_table
        try:
            scored = scorer.score_pairs(read_table(_named_buffer(up)))
            st.dataframe(scored, hide_index=True, width="stretch")
            st.download_button("Download results (.xlsx)", _excel_bytes(scored),
                               file_name="scored_rotations.xlsx")
        except ValueError as e:
            st.error(str(e))

# ------------------------------------------------------------------ tab 4
with tab_model:
    if art:
        m = art["metadata"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hold-out ROC-AUC", f"{m['holdout_metrics']['auc']:.3f}")
        c2.metric("Hold-out PR-AUC", f"{m['holdout_metrics']['pr_auc']:.3f}",
                  help=f"A model that guesses gets {m['baseline_pr_auc']:.3f}")
        c3.metric("Brier score", f"{m['holdout_metrics']['brier']:.3f}")
        c4.metric("Cohort", f"{m['cohort_size']:,}",
                  help=f"{m['positive_rate']:.0%} of these rotations were successful")
        st.markdown(f"**Model** {m['best']['model']} · scaler {m['best']['scaler']} · "
                    f"calibration {m['best']['calibration']} · features used: "
                    f"{', '.join(m['deployed_features'])}")
        st.subheader("Significant attributes")
        st.caption("Likelihood-ratio test per attribute on the development rows; `p_holm` is "
                   "corrected for testing many attributes. `direction` + means higher values go "
                   "with more success. Association, not cause.")
        sig = art["significance"].copy()
        sig["significant"] = sig["p_holm"] < art["cfg"]["experiment"].get("reduced_alpha", 0.05)
        st.dataframe(sig.round(4), hide_index=True, width="stretch")
        st.subheader("What the model leans on")
        imp = art["importance"]
        _hbar(imp, "mean", title="drop in PR-AUC when the attribute is shuffled")
        st.dataframe(imp.round(4), hide_index=True, width="stretch")
        st.caption("`consistent` = positive in every split. Splits overlap, so this is not a "
                   "significance test — use the table above for that.")
        with st.expander("Known limitations"):
            for lim in m["known_limitations"] + m.get("notes", []):
                st.markdown(f"- {lim}")
    st.subheader("Train / retrain")
    st.caption(f"Uses the loaded data and the settings in config.yaml. The model is saved to "
               f"`{model_path}` (personal data — keep local) and an aggregate-only report to "
               f"`{cfg['paths']['report']}` (safe to share).")
    if st.button("Train model", disabled=hr is None):
        log = st.empty()
        msgs = []

        def progress(msg):
            msgs.append(msg)
            log.code("\n".join(msgs))

        try:
            result = train(cfg, hr=hr, progress=progress)
            save_outputs(result, cfg["paths"]["outputs"])
            write_report(result, cfg["paths"]["report"])
            _load_model.clear()
            st.success("Model trained. Reloading…")
            st.rerun()
        except ValueError as e:
            st.error(str(e))
