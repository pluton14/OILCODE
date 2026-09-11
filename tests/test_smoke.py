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


def test_lims_respects_publication_delay():
    """Лабораторный результат нельзя использовать раньше, чем он опубликован.

    Метка времени в ЛИМС — момент отбора пробы, результат появляется в
    системе спустя до 4 часов. Использовать его раньше — утечка из будущего.
    """
    ts = pd.Timestamp("2024-06-15 12:00")
    state = build_process_state(ts)
    for metric, m in state.quality.items():
        if m.source != "LIMS" or m.measured_at is None:
            continue
        available_at = m.measured_at + pd.Timedelta(
            hours=config.LIMS_PUBLICATION_DELAY_H)
        assert available_at <= ts, f"{metric}: результат ещё не был опубликован"


def test_blending_never_violates_product_spec():
    """Любая выданная рецептура обязана проходить спецификацию."""
    from oilcode.agents.blending import BlendingAgent

    result = BlendingAgent().solve({"sulfur_mg_kg": 12.0, "t95_c": 355.0,
                                    "cetane": 50.5})
    if result.feasible:
        b = result.best
        assert b.blended["sulfur_mg_kg"] <= config.PRODUCT_SPEC["sulfur_mg_kg_max"]
        assert b.blended["t95_c"] <= config.PRODUCT_SPEC["t95_max_c"]
        assert b.blended["cetane"] >= config.PRODUCT_SPEC["cetane_min"]
        assert abs(sum(b.fractions.values()) - 1.0) < 1e-6, "доли не дают 100%"
        assert b.improver_pct <= config.HARD_LIMITS["cetane_improver_max_pct"]


def test_blending_refuses_when_feed_too_dirty():
    """Если сырьё слишком грязное, агент обязан отказать, а не выдумать рецепт."""
    from oilcode.agents.blending import BlendingAgent

    result = BlendingAgent().solve({"sulfur_mg_kg": 40.0, "t95_c": 355.0,
                                    "cetane": 50.5})
    assert not result.feasible
    assert result.rejection_reason
