"""Estimators and preprocessing pipelines (notebook §5/§6)."""
from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


class BalancedMLP(MLPClassifier):
    """MLP with balanced class weights, so it is compared fairly with the
    class-weighted logistic regression, random forest and Keras ANN."""

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y)
        with warnings.catch_warnings():
            # expected for some grid candidates; the tuned choice is what counts
            warnings.simplefilter("ignore", ConvergenceWarning)
            return super().fit(X, y, sample_weight=sample_weight)


class KerasANN(ClassifierMixin, BaseEstimator):
    """Same architecture as the original notebook, wrapped as a scikit-learn
    classifier so it fits in a Pipeline and can be saved with joblib."""

    def __init__(self, hidden=(32, 16), dropout=0.3, epochs=200, batch_size=32,
                 patience=15, learning_rate=1e-3, random_state=42):
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        from tensorflow import keras

        keras.utils.set_random_seed(self.random_state)
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        layers = [keras.layers.Input(shape=(X.shape[1],)),
                  keras.layers.Dense(self.hidden[0], activation="relu"),
                  keras.layers.Dropout(self.dropout)]
        layers += [keras.layers.Dense(h, activation="relu") for h in self.hidden[1:]]
        layers += [keras.layers.Dense(1, activation="sigmoid")]
        m = keras.Sequential(layers)
        m.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss="binary_crossentropy")
        w = compute_sample_weight("balanced", y)
        cw = {int(c): float(w[y == c][0]) for c in self.classes_}
        # validation_split takes the LAST rows, so shuffle first
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(len(y))
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience,
                                           restore_best_weights=True)
        m.fit(X[idx], y[idx], validation_split=0.2, epochs=self.epochs,
              batch_size=self.batch_size, class_weight=cw, callbacks=[es], verbose=0)
        self.model_ = m
        return self

    def predict_proba(self, X):
        p = self.model_.predict(np.asarray(X, dtype="float32"), verbose=0).ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # Keras models do not pickle reliably — store them in the native format.
    def __getstate__(self):
        state = self.__dict__.copy()
        if "model_" in state:
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "m.keras")
                state["model_"].save(path)
                with open(path, "rb") as f:
                    state["model_"] = f.read()
        return state

    def __setstate__(self, state):
        if isinstance(state.get("model_"), bytes):
            from tensorflow import keras
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "m.keras")
                with open(path, "wb") as f:
                    f.write(state["model_"])
                state["model_"] = keras.models.load_model(path)
        self.__dict__.update(state)


def resolve_ann_backend(requested: str) -> str:
    if requested in ("auto", "keras"):
        try:
            import tensorflow  # noqa: F401
            return "keras"
        except Exception:
            if requested == "keras":
                raise
    return "sklearn"


# ~150 positives is small for a neural net: let it pick its size and weight
# decay by inner cross-validation on the TRAINING rows only.
ANN_GRID = {"hidden_layer_sizes": [(16,), (32, 16)], "alpha": [1e-3, 1e-1, 1.0]}


def make_estimator(name: str, seed: int, ann_backend: str = "sklearn"):
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                      class_weight="balanced_subsample",
                                      random_state=seed, n_jobs=-1)
    if name == "ann":
        if ann_backend == "keras":
            return KerasANN(random_state=seed)
        return GridSearchCV(BalancedMLP(max_iter=1500, random_state=seed), ANN_GRID,
                            cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                            scoring="average_precision")
    raise ValueError(f"unknown model: {name}")


def make_preprocessor(kind: str, numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    scaler = StandardScaler() if kind == "standard" else MinMaxScaler()
    blocks = []
    if numeric:
        blocks.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                        ("scale", scaler)]), numeric))
    if categorical:
        blocks.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5,
                                  sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(blocks, sparse_threshold=0)


def make_pipeline(name, scaler, seed, numeric, categorical, ann_backend="sklearn") -> Pipeline:
    # The scaler is fitted INSIDE the pipeline, so it only ever sees training
    # rows. This is what prevents leakage — a structure, not a comment.
    return Pipeline([("pre", make_preprocessor(scaler, numeric, categorical)),
                     ("clf", make_estimator(name, seed, ann_backend))])


def prepare_X(features: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Plain float / object columns with np.nan for missing — what sklearn expects."""
    X = pd.DataFrame(index=features.index)
    for c in numeric:
        X[c] = pd.to_numeric(features[c], errors="coerce").astype(float)
    for c in categorical:
        X[c] = pd.Series([str(v) if pd.notna(v) else np.nan for v in features[c]],
                         index=features.index, dtype="object")
    return X
