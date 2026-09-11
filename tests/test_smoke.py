"""Дымовые тесты: цепочка проходит целиком и жёсткие правила соблюдаются."""

import pandas as pd

from oilcode import config
from oilcode.data.state import build_process_state
from oilcode.orchestrator import Orchestrator


def test_state_has_no_future_data():
    """Главное правило: в снимке не должно быть данных из будущего."""
    ts = pd.Timestamp("2024-06-15 12:00")
    state = build_process_state(ts)
    for metric, m in state.quality.items():
        assert m.measured_at is None or m.measured_at <= ts, f"{metric} из будущего"


def test_full_cycle_returns_recommendation():
    rec = Orchestrator().fit().decide("2024-06-15 12:00")
    assert rec.status in ("recommendation", "no_action", "refusal")
    assert rec.explanation, "система обязана объяснять любой ответ"


def test_recommendation_never_violates_sulfur_limit():
    """Предложенное действие не может выводить серу за предел."""
    rec = Orchestrator().fit().decide("2024-06-15 12:00")
    if rec.status == "recommendation":
        sulfur = rec.expected_effect.get("качество", "")
        limit = config.HARD_LIMITS["sulfur_mg_kg_max"]
        value = float(sulfur.split()[1])
        assert value <= limit, f"рекомендация нарушает предел: {value} > {limit}"
