import json
from types import SimpleNamespace

import pandas as pd
import pytest

from bist_quant.kronos_validation import (
    LABEL_UNVALIDATED,
    LABEL_VALIDATED,
    MODEL_BASE,
    MODEL_FINETUNED,
    format_banner,
    validation_status,
    write_validation,
)


def _metrics(skill, direction, model=MODEL_FINETUNED):
    return pd.DataFrame([
        {"model": model, "horizon": "5G", "n": 396, "directional_accuracy_pct": 50.0, "mae_pct": 5.0,
         "naive_mae_pct": 4.5, "skill_vs_naive_pct": -10.0, "information_coefficient": 0.1},
        {"model": model, "horizon": "15G", "n": 396, "directional_accuracy_pct": direction, "mae_pct": 11.0,
         "naive_mae_pct": 8.2, "skill_vs_naive_pct": skill, "information_coefficient": 0.07},
    ])


def _write(tmp_path, skill, direction, fingerprints=None):
    path = str(tmp_path / "validation.json")
    write_validation(_metrics(skill, direction), "2025-11-03", "2026-09-15", fingerprints or {}, path)
    return path


def test_model_without_any_measurement_is_unvalidated(tmp_path):
    status = validation_status(MODEL_FINETUNED, path=str(tmp_path / "yok.json"))

    assert status.is_validated is False
    assert status.label == LABEL_UNVALIDATED
    assert "ölçüm yok" in status.detail


def test_model_that_loses_to_the_naive_forecast_is_unvalidated(tmp_path):
    status = validation_status(MODEL_FINETUNED, path=_write(tmp_path, skill=-34.8, direction=48.2))

    assert status.is_validated is False
    assert "-34.8" in status.detail and "48.2" in status.detail and "396" in status.detail


def test_model_that_beats_naive_and_a_coin_flip_is_validated(tmp_path):
    status = validation_status(MODEL_FINETUNED, path=_write(tmp_path, skill=6.0, direction=56.0))

    assert status.is_validated is True
    assert status.label == LABEL_VALIDATED


def test_beating_naive_error_alone_is_not_enough(tmp_path):
    assert validation_status(MODEL_FINETUNED, path=_write(tmp_path, skill=6.0, direction=49.0)).is_validated is False


def test_measurement_of_another_model_does_not_validate_this_one(tmp_path):
    path = _write(tmp_path, skill=6.0, direction=56.0)

    assert validation_status(MODEL_BASE, path=path).is_validated is False


def test_measurement_taken_before_retraining_is_stale(tmp_path):
    path = _write(tmp_path, skill=6.0, direction=56.0, fingerprints={MODEL_FINETUNED: 100.0})

    fresh = validation_status(MODEL_FINETUNED, path=path, current_fingerprint=100.0)
    stale = validation_status(MODEL_FINETUNED, path=path, current_fingerprint=200.0)

    assert fresh.is_validated is True
    assert stale.is_validated is False
    assert "yeniden eğitildi" in stale.detail


def test_corrupt_file_is_treated_as_no_measurement(tmp_path):
    path = tmp_path / "validation.json"
    path.write_text("{bozuk", encoding="utf-8")

    assert validation_status(MODEL_FINETUNED, path=str(path)).is_validated is False


def test_written_summary_keeps_the_long_horizon_row_per_model(tmp_path):
    path = _write(tmp_path, skill=-34.8, direction=48.2)
    saved = json.loads(open(path, encoding="utf-8").read())

    assert saved["models"][MODEL_FINETUNED]["horizon"] == "15G"
    assert saved["period_start"] == "2025-11-03"


def test_banner_states_the_label_and_the_evidence(tmp_path):
    banner = format_banner(validation_status(MODEL_FINETUNED, path=_write(tmp_path, skill=-34.8, direction=48.2)))

    assert LABEL_UNVALIDATED in banner
    assert "-34.8" in banner


def test_scanner_ranks_with_kronos_only_when_it_is_validated():
    from bist_quant.bist_scanner import should_rank_with_kronos

    validated = SimpleNamespace(predictor=object(), validation=SimpleNamespace(is_validated=True))
    unvalidated = SimpleNamespace(predictor=object(), validation=SimpleNamespace(is_validated=False))

    assert should_rank_with_kronos(validated) is True
    assert should_rank_with_kronos(unvalidated) is False
    assert should_rank_with_kronos(None) is False
