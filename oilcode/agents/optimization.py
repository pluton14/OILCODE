"""Агент оптимизации.

ЗОНА ОТВЕТСТВЕННОСТИ: <впиши имя>

Генерирует варианты изменения режима, выбрасывает недопустимые, сравнивает
оставшиеся и предлагает лучший.

ЧТО ЗДЕСЬ СЧИТАЕТСЯ "ЛУЧШИМ" (важно, это поменялось)
Не "минимум серы". Норма 10 мг/кг относится к товарному продукту после
блендинга, а глубокое обессеривание стоит энергии. Поэтому цель —
МИНИМАЛЬНАЯ СТОИМОСТЬ при выполнении спецификации. Переочистка — убыток,
а не заслуга.

Каждый сценарий проходит два этапа:
  1. изменение режима -> прогноз качества на выходе гидроочистки;
  2. это качество идёт в агент блендинга -> рецептура и её стоимость.
Сценарий допустим, только если блендинг СМОГ свести смесь в спецификацию.

СЕЙЧАС: грубый перебор по одному рычагу за раз. Этого достаточно, чтобы
закрыть требование ТЗ ("несколько вариантов -> отбросить недопустимые ->
сравнить"). Организаторы явно разрешили простую, в том числе линейную модель.

ЧТО ДЕЛАТЬ ДАЛЬШЕ:
  v0  заменить линейную чувствительность на прогноз агента качества
  v1  учесть задержку отклика 0-3 часа (её величину найти по данным)
  v2  Парето-фронт: стоимость против риска для катализатора
  v3  комбинации из двух рычагов одновременно
"""

from __future__ import annotations

from oilcode import config
from oilcode.agents.blending import BlendingAgent
from oilcode.contracts import (
    Change,
    Constraint,
    OptimizationResult,
    ProcessState,
    QualityAssessment,
    ReliabilityAssessment,
    Scenario,
)

# Чувствительность серы на выходе гидроочистки: мг/кг на ОДИН ПРОЦЕНТ
# изменения рычага.
#
# Именно на процент, а не на единицу: теги живут в разных масштабах, и
# "изменение на единицу" означает для них совершенно разное.
#
# ДОПУЩЕНИЕ: знаки взяты из технологической логики, величины оценочные.
# Проверка по данным показала, что линейная корреляция серы с отдельными
# тегами слабая (максимум 0.23), то есть без учёта качества сырья эти
# коэффициенты не восстановить. Это первое, что здесь должно умереть.
SENSITIVITY_PER_PCT = {
    "ht_reactor_temp": -0.80,      # горячее реактор -> глубже обессеривание
    "ht_feed_rate": +0.40,         # больше сырья -> меньше время контакта
    "ht_reactor_pressure": -0.30,  # выше давление -> глубже обессеривание
    "avt_column_top_temp": +0.05,
    "avt_column_bottom_temp": +0.10,
}

# Шаги подобраны операционно правдоподобными: оператор не двигает режим на 10%.
STEPS_PCT = (-0.02, -0.01, 0.0, 0.01, 0.02)


class OptimizationAgent:
    def __init__(self, blending: BlendingAgent | None = None,
                 severity_penalty: float = 0.15):
        self.blending = blending or BlendingAgent()
        # Во сколько условных единиц стоимости обходится единица риска для
        # катализатора. Так риск оборудования попадает в ту же валюту, что и
        # энергозатраты, и их можно честно сравнивать. ДОПУЩЕНИЕ.
        self.severity_penalty = severity_penalty

    def optimize(
        self,
        state: ProcessState,
        quality: QualityAssessment,
        reliability: ReliabilityAssessment,
    ) -> OptimizationResult:
        result = OptimizationResult()

        if state.data_quality.overall == "insufficient":
            result.rejection_reason = (
                "Недостаточно данных: невозможно предсказать последствия "
                "изменения режима"
            )
            return result

        sulfur = state.quality.get(config.TARGET_METRIC)
        if sulfur is None:
            result.rejection_reason = f"Нет значения {config.TARGET_METRIC}"
            return result

        limits_by_tag = {c.tag: c for c in reliability.constraints}
        t95 = state.quality.get("95%.T")
        cetane = state.quality.get("CetaneNumber")

        for name, meta in config.CONTROLS.items():
            tag = meta["tag"]
            current = state.get(config.tag_key(tag, meta["unit"]))
            if current is None or current == 0:
                continue

            for step in STEPS_PCT:
                scenario = self._build_scenario(
                    name, meta, tag, current, step, sulfur.value,
                    t95.value if t95 else None,
                    cetane.value if cetane else None,
                    reliability, limits_by_tag,
                )
                result.scenarios.append(scenario)

        feasible = [s for s in result.scenarios if s.hard_constraints_passed]
        if not feasible:
            result.feasible = False
            result.rejection_reason = (
                "Ни один вариант не проходит: блендинг не сводит смесь в "
                "спецификацию либо нарушены ограничения по оборудованию"
            )
            return result

        # Точка отсчёта: ничего не менять. Все рычаги при шаге 0% дают один
        # и тот же результат, поэтому достаточно любого такого сценария.
        baseline = next((s for s in feasible if s.id.endswith("+0%")), None)
        result.baseline_score = baseline.score if baseline else None

        # score = минус стоимость, поэтому максимум score = минимум затрат.
        best = max(feasible, key=lambda s: s.score)
        result.feasible = True
        result.best_scenario_id = best.id
        return result

    # ----------------------------------------------------------------------

    def _build_scenario(self, name, meta, tag, current, step, sulfur_now,
                        t95_now, cetane_now, reliability, limits_by_tag) -> Scenario:
        proposed = current * (1 + step)
        change = Change(tag=tag, current=round(current, 3), proposed=round(proposed, 3))
        scenario = Scenario(id=f"{name}{step:+.0%}", changes=[change])

        # --- Этап 1: во что превратится качество на выходе гидроочистки ----
        predicted_sulfur = sulfur_now + SENSITIVITY_PER_PCT.get(name, 0.0) * step * 100
        predicted_sulfur = max(predicted_sulfur, 0.1)  # ниже нуля сера не бывает
        scenario.predicted_quality = {config.TARGET_METRIC: round(predicted_sulfur, 2)}
        scenario.predicted_severity = reliability.severity_index
        scenario.energy_proxy_delta = round(step * 100, 1) if "temp" in name else 0.0
        scenario.throughput_delta_pct = round(step * 100, 1) if "feed" in name else 0.0

        # --- Этап 2: сможет ли блендинг свести это в спецификацию ----------
        blend = self.blending.solve({
            "sulfur_mg_kg": predicted_sulfur,
            "t95_c": t95_now if t95_now is not None else 357.0,
            "cetane": cetane_now if cetane_now is not None else 51.0,
        })

        if not blend.feasible:
            scenario.hard_constraints_passed = False
            scenario.rejected_by.append(f"блендинг: {blend.rejection_reason}")
        else:
            scenario.predicted_quality.update(blend.best.blended)
            # Стоимость смеси плюс плата за риск для катализатора.
            total_cost = blend.best.cost + self.severity_penalty * scenario.predicted_severity
            scenario.score = round(-total_cost, 4)

        self._check_equipment_limits(scenario, limits_by_tag, reliability)
        return scenario

    @staticmethod
    def _check_equipment_limits(scenario, limits_by_tag, reliability) -> None:
        """Ограничения от агента надёжности. Нарушил — выбывает."""
        for change in scenario.changes:
            c: Constraint | None = limits_by_tag.get(change.tag)
            if c is None:
                continue
            if c.max is not None and change.proposed > c.max:
                scenario.hard_constraints_passed = False
                scenario.rejected_by.append(
                    f"{change.tag}={change.proposed:.2f} выше предела {c.max:.2f} ({c.reason})"
                )
            if c.min is not None and change.proposed < c.min:
                scenario.hard_constraints_passed = False
                scenario.rejected_by.append(f"{change.tag} ниже предела {c.min:.2f}")

        if not reliability.regime_allowed and any(c.delta > 0 for c in scenario.changes):
            scenario.hard_constraints_passed = False
            scenario.rejected_by.append("агент надёжности запретил ужесточать режим")
