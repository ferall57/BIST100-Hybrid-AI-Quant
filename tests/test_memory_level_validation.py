"""Komite hafızasına yön ile çelişen hedef/stop veya 0-100 dışı güven skoru yazılmaz."""

import pytest

from bist_quant.bist_memory import BistCommitteeMemory
from hybrid_agents.prompts import BIST_PORTFOLIO_MANAGER_PROMPT
from hybrid_agents.verdict_parser import parse_decision_fields


@pytest.fixture
def memory(tmp_path):
    return BistCommitteeMemory(memory_filepath=str(tmp_path / "memory.json"))


def _record(memory, **overrides):
    fields = dict(ticker="THYAO.IS", spot_price=273.5, verdict="SAT", confidence=85,
                  target_1w=270.0, target_15d=260.0, stop_loss=280.0)
    fields.update(overrides)
    return memory.record_decision(**fields)


def test_sell_stop_below_spot_is_not_stored(memory):
    # THYAO_20261016_224423: spot 273.5, stop 271.5 — açığa satışta stop spotun altında olamaz.
    entry = _record(memory, stop_loss=271.5)
    assert entry["stop_loss"] is None


def test_sell_stop_equal_to_target_below_spot_is_not_stored(memory):
    # THYAO_20261007_203233: spot 286.25, hedef 283.99, stop 283.99.
    entry = _record(memory, spot_price=286.25, target_1w=283.99, target_15d=279.86, stop_loss=283.99)
    assert entry["stop_loss"] is None
    assert entry["target_1w"] == 283.99


def test_sell_targets_above_spot_are_not_stored(memory):
    entry = _record(memory, target_1w=275.0, target_15d=290.0)
    assert entry["target_1w"] is None
    assert entry["target_15d"] is None


def test_consistent_sell_levels_are_kept(memory):
    entry = _record(memory)
    assert (entry["target_1w"], entry["target_15d"], entry["stop_loss"]) == (270.0, 260.0, 280.0)


def test_buy_levels_on_the_wrong_side_are_not_stored(memory):
    entry = _record(memory, verdict="AL", target_1w=270.0, target_15d=290.0, stop_loss=280.0)
    assert entry["target_1w"] is None
    assert entry["target_15d"] == 290.0
    assert entry["stop_loss"] is None


def test_hold_levels_are_stored_without_direction_check(memory):
    entry = _record(memory, verdict="TUT", target_1w=270.0, target_15d=280.0, stop_loss=271.5)
    assert (entry["target_1w"], entry["target_15d"], entry["stop_loss"]) == (270.0, 280.0, 271.5)


@pytest.mark.parametrize("confidence", [419, 101, -5])
def test_confidence_outside_0_100_is_not_stored(memory, confidence):
    # BIMAS_20260906_001120 güven skorunu 419 olarak saklamıştı.
    entry = _record(memory, verdict="AL", spot_price=415.75, target_1w=428.22, target_15d=449.01,
                    stop_loss=399.12, confidence=confidence)
    assert entry["confidence"] is None


def test_validation_also_applies_when_a_recent_record_is_updated(memory):
    _record(memory)
    updated = _record(memory, stop_loss=271.5, confidence=419)
    assert len(memory.memory_data["history"]) == 1
    assert updated["stop_loss"] is None
    assert updated["confidence"] is None


def test_manager_prompt_states_stop_direction_for_both_verdicts():
    stop_line = next(line for line in BIST_PORTFOLIO_MANAGER_PROMPT.splitlines() if "Stop Seviyesi" in line)
    assert "SAT" in stop_line and "ÜSTÜNDE" in stop_line
    assert "AL" in stop_line and "ALTINDA" in stop_line


def test_parser_still_reads_the_stop_line_of_the_manager_template():
    stop_line = next(line for line in BIST_PORTFOLIO_MANAGER_PROMPT.splitlines() if "Stop Seviyesi" in line)
    filled = stop_line.replace("[Fiyat]", "280.00")
    assert parse_decision_fields(filled, 273.5)["stop_loss"] == 280.0
