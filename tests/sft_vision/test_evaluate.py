"""Tests for multimodal SFT evaluation (pure parsing + scoring, stubbed predictor)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from geap_tuning.sft_vision.data import LABEL_MAP, build_image_records
from geap_tuning.sft_vision.data import SelectedImage as _SelectedImage
from geap_tuning.sft_vision.evaluate import (
    UNKNOWN,
    gold_label,
    image_gcs_uri_of,
    parse_prediction,
    resolve_local_path,
    run_image_eval,
    select_best_experiment,
)

if TYPE_CHECKING:
    from geap_tuning.schemas import Record


def _record(class_name: str, filename: str) -> Record:
    item = _SelectedImage(
        split="test",
        class_name=class_name,
        filename=filename,
        local_path=Path("/tmp") / filename,  # noqa: S108 - label only, not read
        mime_type="image/jpeg",
    )
    return build_image_records([item], bucket="gs://bucket", gcs_prefix="prefix")[0]


def test_parse_prediction_matches_key_or_label() -> None:
    assert parse_prediction("This looks like gingivitis to me") == LABEL_MAP["gingivitis"]
    assert parse_prediction("gingivitis") == "Gingivitis"
    assert parse_prediction("I think it is Dental Calculus.") == "Dental Calculus"
    assert parse_prediction("clearly oral cancer") == "Oral Cancer"


def test_parse_prediction_unknown_fallback() -> None:
    assert parse_prediction("no idea what this is") == UNKNOWN
    assert parse_prediction("") == UNKNOWN


def test_gold_label_and_uri_extraction() -> None:
    record = _record("caries", "caries_1.jpg")
    assert gold_label(record) == LABEL_MAP["caries"]
    assert image_gcs_uri_of(record) == "gs://bucket/prefix/data/test/caries/caries_1.jpg"


def test_resolve_local_path_maps_back_under_root() -> None:
    uri = "gs://bucket/prefix/data/test/caries/caries_1.jpg"
    resolved = resolve_local_path(uri, "/local/root")
    assert resolved == Path("/local/root/test/caries/caries_1.jpg")


def test_resolve_local_path_rejects_non_dataset_uri() -> None:
    with pytest.raises(ValueError, match="not a staged dataset image"):
        resolve_local_path("gs://bucket/other/file.jpg", "/local/root")


def test_run_image_eval_scores_against_gold() -> None:
    records = [_record("caries", "a.jpg"), _record("gingivitis", "b.jpg")]
    # Predict the true label for the first, a wrong label for the second.
    predictions = {
        image_gcs_uri_of(records[0]): "Dental Caries",
        image_gcs_uri_of(records[1]): "Oral Cancer",
    }
    metrics = run_image_eval(records, lambda r: predictions[image_gcs_uri_of(r)])
    assert metrics["accuracy"] == 0.5
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert "report" in metrics


def test_run_image_eval_all_correct() -> None:
    records = [_record("ulcer", "a.jpg")]
    metrics = run_image_eval(records, lambda _r: "oral ulcer")
    assert metrics["accuracy"] == 1.0


def test_select_best_experiment_picks_max_accuracy() -> None:
    results = {
        "baseline": {"accuracy": 0.7, "macro_f1": 0.6},
        "wide": {"accuracy": 0.9, "macro_f1": 0.8},
    }
    assert select_best_experiment(results) == "wide"


def test_select_best_experiment_tie_breaks_by_name() -> None:
    results = {
        "zeta": {"accuracy": 0.8},
        "alpha": {"accuracy": 0.8},
    }
    assert select_best_experiment(results) == "alpha"


def test_select_best_experiment_empty_raises() -> None:
    with pytest.raises(ValueError, match="No experiment results"):
        select_best_experiment({})
