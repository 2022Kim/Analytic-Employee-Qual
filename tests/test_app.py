"""Smoke test of the webapp with synthetic data (no browser needed)."""
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def test_app_demo_train_and_score():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Synthetic demo data").run()
    assert not at.exception
    train_btn = next(b for b in at.button if b.label == "Train model")
    train_btn.click().run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]

    estimate = next(b for b in at.button if b.label == "Estimate")
    estimate.click().run()
    assert not at.exception, at.exception
    assert any("success probability" in m.label for m in at.metric)

    rank = next(b for b in at.button if b.label == "Rank")
    rank.click().run()
    assert not at.exception, at.exception
