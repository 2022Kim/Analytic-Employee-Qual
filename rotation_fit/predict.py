"""Score hypothetical rotations: employee x destination role -> success probability."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import HRData, key
from .features import Roster
from .models import prepare_X

PAIR_COLUMNS = ["NIK", "destination_position", "destination_org_unit"]
OPTIONAL_PAIR_COLUMNS = ["rotation_date", "destination_group"]


def save_artifact(artifact: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_artifact(path) -> dict:
    art = joblib.load(path)
    if not isinstance(art, dict) or art.get("version", 0) < 2:
        raise ValueError("model file is from an older notebook version — retrain it "
                         "with `python -m rotation_fit train`")
    return art


class RotationScorer:
    def __init__(self, artifact: dict, hr: HRData):
        self.art = artifact
        self.hr = hr
        self.cfg = artifact["cfg"]
        self.fb = artifact["feature_builder"]
        self.roster = Roster(hr.career, hr.emp["_k"])

    @classmethod
    def from_path(cls, path, hr: HRData) -> "RotationScorer":
        return cls(load_artifact(path), hr)

    # ------------------------------------------------------------------ core
    def _pairs(self, pairs: pd.DataFrame) -> pd.DataFrame:
        p = pairs.copy()
        p.columns = [str(c).strip() for c in p.columns]
        if "_k" not in p.columns:
            if "NIK" not in p.columns:
                raise ValueError(f"pairs need columns {PAIR_COLUMNS}")
            p["_k"] = key(p["NIK"])
        missing = [c for c in PAIR_COLUMNS[1:] if c not in p.columns]
        if missing:
            raise ValueError(f"pairs are missing column(s): {missing}")
        if "rotation_date" not in p.columns:
            p["rotation_date"] = pd.Timestamp.today().normalize()
        p["rotation_date"] = pd.to_datetime(p["rotation_date"]).fillna(
            pd.Timestamp.today().normalize())
        return p

    def features(self, pairs: pd.DataFrame) -> pd.DataFrame:
        return self.fb.transform(self.hr, self._pairs(pairs), self.cfg["ratings"]["current"])

    def _proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.art["model"].predict_proba(X[self.art["features"]])[:, 1]

    def score_pairs(self, pairs: pd.DataFrame) -> pd.DataFrame:
        f = self.features(pairs)
        X = prepare_X(f, self.art["numeric"], self.art["categorical"])
        p = self._proba(X)
        out = pd.DataFrame({
            "NIK": f["_k"], "Nama": f.get("Nama"),
            "origin_position": f["origin_position"],
            "destination_position": f["destination_position"],
            "destination_org_unit": f["destination_org_unit"],
            "rotation_date": f["rotation_date"].dt.date,
            "success_pct": np.round(p * 100, 1),
            "out_of_distribution": self.art["ood"].flag(X),
            "similarity_basis": f["similarity_fallback_level"],
        })
        for c in self.art["features"]:
            out[c] = f[c].to_numpy()
        return out

    def rank_candidates(self, destination_position: str, destination_org_unit: str,
                        candidates=None, date=None) -> pd.DataFrame:
        """Score every candidate (default: all employees) for one destination role."""
        keys = list(self.hr.emp["_k"]) if candidates is None else [str(k) for k in candidates]
        pairs = pd.DataFrame({"_k": keys, "destination_position": destination_position,
                              "destination_org_unit": destination_org_unit,
                              "rotation_date": pd.Timestamp(date) if date else
                              pd.Timestamp.today().normalize()})
        out = self.score_pairs(pairs)
        # someone already in the destination role is not a rotation candidate
        same = out["origin_position"].astype("string").str.casefold().str.strip() == \
            str(destination_position).casefold().strip()
        out = out[~same.fillna(False)]
        return out.sort_values("success_pct", ascending=False).reset_index(drop=True)

    def explain(self, pair: pd.DataFrame) -> pd.DataFrame:
        """What-if drivers for ONE pair: change in probability when a feature is
        reset to the typical training value. Positive = this attribute helps."""
        f = self.features(pair.head(1))
        X = prepare_X(f, self.art["numeric"], self.art["categorical"])
        base = float(self._proba(X)[0])
        rows = []
        for c in self.art["features"]:
            Xc = X.copy()
            Xc[c] = self.art["baseline"][c]
            rows.append({"feature": c, "value": X[c].iloc[0],
                         "typical_value": self.art["baseline"][c],
                         "effect_pct_points": round((base - float(self._proba(Xc)[0])) * 100, 1)})
        return (pd.DataFrame(rows)
                .sort_values("effect_pct_points", key=np.abs, ascending=False)
                .reset_index(drop=True))

    # ------------------------------------------------------------ helpers
    def destination_units(self) -> list[str]:
        return self.roster.units()

    def destination_positions(self) -> list[str]:
        return self.roster.positions()
