"""Which attributes are statistically significant?

Drop-one likelihood-ratio tests on a logistic regression: for each attribute,
compare the model with and without it. The deviance difference follows a
chi-square distribution with df = number of model columns the attribute adds
(1 for a numeric attribute, levels - 1 for a categorical one). Holm's
correction keeps the chance of ANY false "significant" at alpha.

Permutation importance answers "what does the model lean on"; this answers
"is there evidence the attribute is associated with success at all". Neither
is causal.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression

from .models import make_preprocessor


def _loglik(Z: np.ndarray, y: np.ndarray) -> tuple[float, LogisticRegression]:
    # C very large = effectively unpenalised, but stays finite under separation
    m = LogisticRegression(C=1e6, max_iter=5000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Z, y)
    p = np.clip(m.predict_proba(Z)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))), m


def _design(X, num, cat):
    if not num and not cat:
        return np.zeros((len(X), 0)), None
    pre = make_preprocessor("standard", num, cat).fit(X)
    return pre.transform(X), pre


def _rank(Z):
    return np.linalg.matrix_rank(np.column_stack([np.ones(len(Z)), Z])) if Z.size else 1


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m = len(p)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[k]))
        adj[k] = running
    return pd.Series(adj)[p.index]


def lr_tests(X: pd.DataFrame, y: np.ndarray, num: list, cat: list) -> pd.DataFrame:
    y = np.asarray(y)
    Z_full, pre = _design(X, num, cat)
    ll_full, model = _loglik(Z_full, y)
    r_full = _rank(Z_full)
    names = list(pre.get_feature_names_out()) if pre is not None else []
    coefs = dict(zip(names, model.coef_.ravel()))
    rows = []
    for a in num + cat:
        n2, c2 = [c for c in num if c != a], [c for c in cat if c != a]
        Z_red, _ = _design(X, n2, c2)
        ll_red = _loglik(Z_red, y)[0] if Z_red.size else float(
            np.sum(y * np.log(y.mean()) + (1 - y) * np.log(1 - y.mean())))
        df = max(r_full - _rank(Z_red), 1)
        stat = max(2 * (ll_full - ll_red), 0.0)
        row = {"feature": a, "lr_stat": stat, "df": int(df), "p_value": float(chi2.sf(stat, df))}
        if a in num:
            b = coefs.get(f"num__{a}", np.nan)
            row["direction"] = "+" if b > 0 else "-"
            row["odds_ratio_per_sd"] = float(np.exp(b))
        else:
            row["direction"] = "by category"
            row["odds_ratio_per_sd"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.set_index("feature")["p_value"]).to_numpy()
    return out.sort_values("p_value").reset_index(drop=True)


def select_significant(X, y, num, cat, alpha: float, max_features: int) -> list:
    """Holm-significant attributes (at most `max_features`, strongest first).
    Falls back to the single strongest attribute when none pass."""
    t = lr_tests(X, y, num, cat)
    sig = list(t.loc[t["p_holm"] < alpha, "feature"])[:max_features]
    return sig or list(t["feature"].head(1))
