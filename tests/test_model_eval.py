import numpy as np
import pytest

from bist_quant.bist_model_eval import forecast_metrics, forecast_origins, paired_comparison

ACTUAL = np.array([4.0, -2.0, 6.0, -8.0, 1.0, -3.0])


def test_perfect_forecast_scores_full_marks():
    metrics = forecast_metrics(ACTUAL, ACTUAL)

    assert metrics["n"] == 6
    assert metrics["directional_accuracy_pct"] == 100.0
    assert metrics["mae_pct"] == 0.0
    assert metrics["skill_vs_naive_pct"] == pytest.approx(100.0)
    assert metrics["information_coefficient"] == pytest.approx(1.0)


def test_predicting_no_change_has_zero_skill_over_the_naive_baseline():
    metrics = forecast_metrics(np.zeros_like(ACTUAL), ACTUAL)

    assert metrics["mae_pct"] == pytest.approx(metrics["naive_mae_pct"])
    assert metrics["skill_vs_naive_pct"] == pytest.approx(0.0)


def test_opposite_forecast_is_worse_than_naive():
    metrics = forecast_metrics(-ACTUAL, ACTUAL)

    assert metrics["directional_accuracy_pct"] == 0.0
    assert metrics["skill_vs_naive_pct"] < 0.0
    assert metrics["information_coefficient"] == pytest.approx(-1.0)


def test_constant_forecast_has_no_information_coefficient():
    assert forecast_metrics(np.full(6, 2.0), ACTUAL)["information_coefficient"] == 0.0


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        forecast_metrics(np.zeros(3), ACTUAL)


def test_paired_comparison_counts_how_often_the_first_model_is_closer():
    errors_a = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    errors_b = errors_a + 2.0

    result = paired_comparison(errors_a, errors_b)

    assert result["first_closer_pct"] == 100.0
    assert result["mean_abs_error_diff"] == pytest.approx(-2.0)
    assert result["p_value"] < 0.01


def test_paired_comparison_of_equal_models_is_not_significant():
    errors = np.array([1.0, 2.0, 3.0, 4.0])

    result = paired_comparison(errors, errors[::-1].copy())

    assert result["first_closer_pct"] == 50.0
    assert result["p_value"] == pytest.approx(1.0)


def test_ties_are_excluded_from_the_sign_test():
    result = paired_comparison(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))

    assert result["p_value"] == 1.0
    assert result["first_closer_pct"] is None


def test_origins_step_back_from_the_last_bar_that_still_has_a_full_horizon():
    origins = forecast_origins(n_bars=500, horizon=15, step=20, count=3, min_index=256)

    assert origins == [444, 464, 484]
    assert all(origin + 15 <= 499 for origin in origins)


def test_origins_never_reach_before_the_minimum_index():
    assert forecast_origins(n_bars=300, horizon=15, step=20, count=5, min_index=256) == [264, 284]


def test_no_origins_when_history_is_too_short():
    assert forecast_origins(n_bars=200, horizon=15, step=20, count=5, min_index=256) == []
