"""Feature engineering (notebook §3 EQS and §4 similarity), reusable for scoring.

The same `FeatureBuilder` produces the training table and the features of a
hypothetical rotation (employee x destination role), so training and the
webapp can never drift apart.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .cohort import rating_ordinal
from .data import HRData, norm, norm_value

EQS_COLS = ["eqs_kinerja", "eqs_kompetensi", "eqs_rekam_jejak",
            "eqs_pelatihan", "eqs_sertifikasi", "eqs_aspirasi"]
NUMERIC_CANDIDATES = EQS_COLS + ["eqs_pengalaman", "usia", "pendidikan_years",
                                 "similarity_score"]
CATEGORICAL_CANDIDATES = ["Level Jabatan", "job_family_changed", "gender", "marital"]

EDU_YEARS = {
    "sma": 12, "smu": 12, "smk": 12, "stm": 12, "slta": 12, "ma": 12, "paket c": 12,
    "d1": 13, "d2": 14, "d3": 15, "d4": 16, "s1": 16, "profesi": 17, "s2": 18, "s3": 21,
}
# whole words only: plain substring matching read "S2 Manajemen" as "ma" (12 years)
_EDU_PATTERNS = [(re.compile(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])"), v)
                 for k, v in EDU_YEARS.items()]


def edu_years(v) -> float:
    """Years of schooling; the highest level mentioned wins ("D3 / S1" -> 16)."""
    s = norm_value(v)
    found = [years for pat, years in _EDU_PATTERNS if pat.search(s)]
    return float(max(found)) if found else np.nan


def rekam_jejak(aktif: pd.Series, non_aktif: pd.Series) -> np.ndarray:
    """0 = active sanction, 50 = past record, 100 = clean (blank = no record)."""
    def present(s):
        return s.notna() & (s.astype("string").str.strip().fillna("") != "")
    return np.where(present(aktif), 0, np.where(present(non_aktif), 50, 100))


def text_blob(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    parts = df[cols].astype("string").apply(lambda c: c.str.replace("\n", " ").str.strip())
    return parts.apply(lambda r: " ; ".join(v for v in r if pd.notna(v) and v), axis=1)


def aspirasi_score(row: pd.Series, target: str, points: dict, threshold: float) -> int:
    """Score by WHICH column names the destination role; the highest wins."""
    tgt = set(norm_value(target).split())
    if not tgt:
        return 0
    best = 0
    for col, pts in points.items():
        val = row.get(col)
        if val is None or (not isinstance(val, str) and pd.isna(val)):
            continue
        for line in str(val).split("\n"):
            words = set(line.strip().lstrip("-").strip().casefold().split())
            if words and len(words & tgt) / len(tgt) >= threshold:
                best = max(best, pts)
    return best


class RelevanceModel:
    """TF-IDF relevance of training/certification text to the target position.

    Fitted once on the whole HR corpus (no labels involved) and stored with the
    model, so a single new pair is scored with the same vocabulary.
    """

    def fit(self, corpus: list[str]):
        corpus = [c for c in corpus if c and c.strip()]
        self.vec_ = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
        if corpus:
            self.vec_.fit(corpus)
        else:
            self.vec_ = None
        return self

    def similarity(self, texts: pd.Series, targets: pd.Series) -> np.ndarray:
        if self.vec_ is None:
            return np.zeros(len(texts))
        A = self.vec_.transform(texts.fillna("").astype(str))
        B = self.vec_.transform(targets.fillna("").astype(str))
        # rows are L2-normalised, so the row-wise dot product is the cosine
        return np.asarray(A.multiply(B).sum(axis=1)).ravel()

    def flags(self, texts: pd.Series, targets: pd.Series, threshold: float) -> np.ndarray:
        out = (self.similarity(texts, targets) >= threshold).astype(int)
        out[texts.fillna("").astype(str).str.strip().to_numpy() == ""] = 0
        return out


class Roster:
    """Who sat in which org unit / position on a given date, from the career file.

    The dataset holds the POST-September-2025 structure, so historical team
    membership has to be reconstructed from the transactions.
    """

    def __init__(self, career: pd.DataFrame, emp_keys):
        src = career[career["_k"].isin(set(emp_keys))].dropna(subset=["Start Date"])
        src = src.sort_values("Start Date", kind="stable")
        self._src = pd.DataFrame({
            "_k": src["_k"].to_numpy(), "date": src["Start Date"].to_numpy(),
            "unit": norm(src["Organization Unit"]).to_numpy(),
            "unit_raw": src["Organization Unit"].astype("string").str.strip().to_numpy(),
            "position": src["Position"].astype("string").str.strip().to_numpy(),
        })
        self._cache: dict = {}

    def as_of(self, date) -> pd.DataFrame:
        d = pd.Timestamp(date).normalize()
        if d not in self._cache:
            snap = self._src[self._src["date"] <= d].groupby("_k").tail(1).set_index("_k")
            groups = {u: list(g.index) for u, g in snap.groupby("unit")}
            self._cache[d] = (snap, groups)
        return self._cache[d][0]

    def members(self, unit: str, date) -> list:
        self.as_of(date)
        return self._cache[pd.Timestamp(date).normalize()][1].get(norm_value(unit), [])

    def units(self, date=None) -> list[str]:
        snap = self.as_of(date if date is not None else pd.Timestamp.today())
        return sorted(snap["unit_raw"].dropna().unique())

    def positions(self) -> list[str]:
        return sorted(pd.Series(self._src["position"]).dropna().unique())


class PercentileScaler:
    """Maps a raw value to 1-100 against the training distribution (Tabel 3)."""

    def fit(self, values):
        v = np.asarray(values, dtype=float)
        self.ref_ = np.sort(v[~np.isnan(v)])
        return self

    def transform(self, values) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        out = np.full(v.shape, np.nan)
        if not len(self.ref_):
            return out
        ok = ~np.isnan(v)
        lo = np.searchsorted(self.ref_, v[ok], side="left")
        hi = np.searchsorted(self.ref_, v[ok], side="right")
        pct = (lo + hi) / 2 / len(self.ref_)     # mid-rank, like rank(pct=True)
        out[ok] = np.round(pct * 99 + 1, 1)
        return out


def gower_similarity(a_num, B_num, ranges, a_cat, B_cat) -> float:
    """1 - mean Gower distance from one row to many rows. NaNs are skipped."""
    parts = []
    if a_num.size:
        parts.append(np.abs(B_num - a_num) / np.where(ranges == 0, 1, ranges))
    if a_cat.size:
        a = np.broadcast_to(a_cat, B_cat.shape)
        missing = pd.isna(B_cat) | pd.isna(a)
        d = (np.where(missing, "", B_cat).astype(str) != np.where(missing, "", a).astype(str))
        d = d.astype(float)
        d[missing] = np.nan
        parts.append(d)
    if not parts:
        return np.nan
    D = np.hstack(parts)
    if np.isnan(D).all():
        return np.nan
    return 1 - np.nanmean(D)


class FeatureBuilder:
    """Fit on the training HR snapshot, then transform any employee x role pairs.

    `pairs` needs: _k, rotation_date, destination_position, destination_org_unit.
    Optional: origin_position (else taken from the roster), destination_group.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def __getstate__(self):
        # Never pickle the HR snapshot into the model file — it is rebuilt from
        # whatever data is loaded at scoring time.
        state = self.__dict__.copy()
        state.pop("attrs_", None)
        state.pop("_career_by_k", None)
        return state

    # ---------------------------------------------------------- helpers
    def _attrs(self, hr: HRData) -> pd.DataFrame:
        e = hr.emp.set_index("_k")
        born = pd.to_datetime(e["Tgl Lahir"], errors="coerce")
        a = pd.DataFrame(index=e.index)
        # Age in years at a FIXED reference. Gower uses differences only, so the
        # reference date cancels out; per-pair age is computed separately.
        a["usia"] = (born - pd.Timestamp("2000-01-01")).dt.days / 365.25
        a["pendidikan_years"] = e["Level Pendidikan 1"].map(edu_years)
        a["job_fam_static"] = e["Job Fam"].astype("string").str.strip()
        a["group"] = e["Group"].astype("string").str.strip()
        for c in ("gender", "marital"):
            a[c] = e[c].astype("string").str.strip() if c in e.columns else pd.NA
        return a

    def _job_family(self, hr: HRData, keys, positions) -> pd.Series:
        static = self.attrs_.reindex(keys)["job_fam_static"]
        if not hr.jf_map:
            return pd.Series(static.to_numpy(), index=keys, dtype="object")
        mapped = norm(pd.Series(positions, index=keys)).map(hr.jf_map)
        return mapped.astype("object").where(mapped.notna(), static.to_numpy())

    def _sim_dims(self, hr: HRData):
        dims = self.cfg["similarity"]["dims"]
        num = [d for d in dims if d in ("usia", "pendidikan_years")
               and self.attrs_[d].notna().any()]
        cat = [d for d in dims if d in ("gender", "marital") and self.attrs_[d].notna().any()]
        if "job_family" in dims:
            cat.append("job_family")
        return num, cat

    def _pengalaman(self, hr: HRData, k, date, dest_jf) -> float:
        """Years spent in the destination job family before the rotation."""
        if not hr.jf_map or pd.isna(dest_jf):
            return np.nan
        rows = self._career_by_k.get(k)
        if rows is None or rows.empty:
            return 0.0
        rows = rows[rows["Start Date"] < date]
        if rows.empty:
            return 0.0
        starts = rows["Start Date"].to_numpy()
        ends = np.append(starts[1:], np.datetime64(pd.Timestamp(date)))
        jf = norm(rows["Position"]).map(hr.jf_map).to_numpy()
        days = (ends - starts).astype("timedelta64[D]").astype(float)
        return float(days[jf == dest_jf].sum() / 365.25)

    # ---------------------------------------------------------- fit / transform
    def fit(self, hr: HRData, pairs: pd.DataFrame, rating_col: str):
        self.fit_transform(hr, pairs, rating_col)
        return self

    def fit_transform(self, hr: HRData, pairs: pd.DataFrame, rating_col: str) -> pd.DataFrame:
        e = self.cfg["eqs"]
        self.attrs_ = self._attrs(hr)
        jf = pd.to_numeric(hr.emp["Job Fit"], errors="coerce")
        # Job Fit is a 0-1 ratio in the source; accept a 0-100 export too.
        self.job_fit_multiplier_ = 1.0 if jf.median() > 1.5 else 100.0

        corpus = (list(text_blob(hr.emp, e["pelatihan_columns"]))
                  + list(text_blob(hr.emp, e["sertifikasi_columns"]))
                  + list(hr.career["Position"].dropna().astype(str).unique()))
        self.relevance_ = RelevanceModel().fit(corpus)

        num, _ = self._sim_dims(hr)
        A = self.attrs_[num].to_numpy(dtype=float) if num else np.empty((len(self.attrs_), 0))
        self.ranges_ = (np.nanmax(A, axis=0) - np.nanmin(A, axis=0)) if num else np.array([])
        self.sim_scaler_ = PercentileScaler().fit([])
        feats = self.transform(hr, pairs, rating_col)
        self.sim_scaler_ = PercentileScaler().fit(feats["similarity_raw"])
        feats["similarity_score"] = self.sim_scaler_.transform(feats["similarity_raw"])
        return feats

    def transform(self, hr: HRData, pairs: pd.DataFrame, rating_col: str) -> pd.DataFrame:
        cfg, e = self.cfg, self.cfg["eqs"]
        self.attrs_ = self._attrs(hr)          # refresh to the current snapshot
        self._career_by_k = {k: g.sort_values("Start Date", kind="stable")
                             for k, g in hr.career.dropna(subset=["Start Date"]).groupby("_k")}
        roster = Roster(hr.career, hr.emp["_k"])

        p = pairs.reset_index(drop=True).copy()
        p["rotation_date"] = pd.to_datetime(p["rotation_date"])
        unknown = sorted(set(p["_k"]) - set(hr.emp["_k"]))
        if unknown:
            raise ValueError(f"{len(unknown)} employee ID(s) not found in the dataset, "
                             f"e.g. {unknown[:3]}")
        emp_cols = (["_k", "Nama", rating_col, "Job Fit", "Tgl Lahir", "Level Pendidikan 1",
                     "Level Jabatan", "Group", "Divisi",
                     "Catatan Disiplin (Aktif)", "Catatan Disiplin (Non-aktif)"]
                    + e["pelatihan_columns"] + e["sertifikasi_columns"]
                    + list(e["aspirasi_points"])
                    + [c for c in ("gender", "marital") if c in hr.emp.columns])
        emp_cols = list(dict.fromkeys(c for c in emp_cols if c not in p.columns or c == "_k"))
        f = p.merge(hr.emp[emp_cols], on="_k", how="left")

        if "origin_position" not in f.columns:
            f["origin_position"] = pd.NA
        missing_origin = f["origin_position"].isna()
        if missing_origin.any():
            f.loc[missing_origin, "origin_position"] = [
                roster.as_of(d)["position"].get(k, pd.NA)
                for k, d in zip(f.loc[missing_origin, "_k"], f.loc[missing_origin, "rotation_date"])]

        # ---- 1. Kinerja (0-120) — pre-rotation rating only ----------
        f["eqs_kinerja"] = rating_ordinal(f[rating_col], cfg).map(e["kinerja_points"])
        # ---- 2. Kesesuaian kompetensi (0-100) — Job Fit -------------
        f["eqs_kompetensi"] = (pd.to_numeric(f["Job Fit"], errors="coerce")
                               * self.job_fit_multiplier_).clip(0, 100)
        # ---- 3. Rekam jejak (0 / 50 / 100) --------------------------
        f["eqs_rekam_jejak"] = rekam_jejak(f["Catatan Disiplin (Aktif)"],
                                           f["Catatan Disiplin (Non-aktif)"])
        # ---- 4/5. Pelatihan and Sertifikasi relevance to the TARGET role
        f["eqs_pelatihan"] = self.relevance_.flags(
            text_blob(f, e["pelatihan_columns"]), f["destination_position"],
            e["tfidf_threshold"]) * 100
        f["eqs_sertifikasi"] = self.relevance_.flags(
            text_blob(f, e["sertifikasi_columns"]), f["destination_position"],
            e["tfidf_threshold"]) * 100
        # ---- 6. Aspirasi (0 / 20 / 25 / 30) -------------------------
        f["eqs_aspirasi"] = [aspirasi_score(r, r["destination_position"], e["aspirasi_points"],
                                            e["aspirasi_match_threshold"])
                             for _, r in f.iterrows()]

        # ---- demographics -------------------------------------------
        f["usia"] = (f["rotation_date"] - pd.to_datetime(f["Tgl Lahir"], errors="coerce")
                     ).dt.days / 365.25
        f["pendidikan_years"] = f["Level Pendidikan 1"].map(edu_years)

        # ---- job family ---------------------------------------------
        if hr.jf_map:
            f["origin_job_family"] = norm(f["origin_position"]).map(hr.jf_map)
            f["destination_job_family"] = norm(f["destination_position"]).map(hr.jf_map)
            both = f[["origin_job_family", "destination_job_family"]].notna().all(axis=1)
            f["job_family_changed"] = np.where(
                both, (f["origin_job_family"] != f["destination_job_family"]).astype(str),
                None)
        else:
            f["origin_job_family"] = f["destination_job_family"] = np.nan
            f["job_family_changed"] = None

        # ---- 7. Pengalaman — years in destination job family ---------
        f["eqs_pengalaman"] = [self._pengalaman(hr, k, d, jf) for k, d, jf in
                               zip(f["_k"], f["rotation_date"], f["destination_job_family"])]

        # ---- similarity to the destination team ---------------------
        raw, level, size = self._similarity(hr, f, roster)
        f["similarity_raw"] = raw
        f["similarity_fallback_level"] = level
        f["team_size"] = size
        f["similarity_score"] = self.sim_scaler_.transform(raw)
        return f

    def _similarity(self, hr: HRData, f: pd.DataFrame, roster: Roster):
        num, cat = self._sim_dims(hr)
        min_team = self.cfg["similarity"]["min_team_size"]
        attrs = self.attrs_
        by_group = {g: list(ix) for g, ix in attrs.groupby("group").groups.items()}
        raw, level, size = [], [], []
        for _, row in f.iterrows():
            k, d = row["_k"], row["rotation_date"]
            members = [m for m in roster.members(row["destination_org_unit"], d) if m != k]
            lvl = "destination_org_unit"
            if len(members) < min_team:
                grp = row.get("destination_group")
                if grp is None or pd.isna(grp):
                    member_groups = attrs.reindex(members)["group"].dropna()
                    grp = member_groups.mode().iloc[0] if len(member_groups) else None
                if grp is not None:
                    members = [m for m in by_group.get(grp, []) if m != k]
                    lvl = "Group"
            if len(members) < min_team:
                raw.append(np.nan)
                level.append("insufficient")
                size.append(len(members))
                continue
            snap = roster.as_of(d)
            a_num = attrs.loc[[k], num].to_numpy(dtype=float)
            B_num = attrs.loc[members, num].to_numpy(dtype=float)
            cat_vals_a, cat_vals_B = [], []
            for c in cat:
                if c == "job_family":
                    cat_vals_a.append(self._job_family(hr, [k], [row["origin_position"]]).to_numpy())
                    pos = snap["position"].reindex(members).to_numpy()
                    cat_vals_B.append(self._job_family(hr, members, pos).to_numpy())
                else:
                    cat_vals_a.append(attrs.loc[[k], c].to_numpy(dtype=object))
                    cat_vals_B.append(attrs.loc[members, c].to_numpy(dtype=object))
            a_cat = np.column_stack(cat_vals_a).astype(object) if cat else np.empty((1, 0))
            B_cat = np.column_stack(cat_vals_B).astype(object) if cat else np.empty((len(members), 0))
            raw.append(gower_similarity(a_num, B_num, self.ranges_, a_cat, B_cat))
            level.append(lvl)
            size.append(len(members))
        return raw, level, size
