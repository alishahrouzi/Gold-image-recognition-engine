"""Regression tests for the S4.9 selected MVP model runtime contract."""

from __future__ import annotations

from inference.gallery import DEFAULT_SIAMESE_MODEL, SELECTED_MVP_MODEL


def test_runtime_default_matches_s4_9_selected_model() -> None:
    assert SELECTED_MVP_MODEL == "s3.5_cross_category"
    assert DEFAULT_SIAMESE_MODEL == SELECTED_MVP_MODEL
