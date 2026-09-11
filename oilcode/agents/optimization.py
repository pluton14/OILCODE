"""Агент оптимизации.

ЗОНА ОТВЕТСТВЕННОСТИ: <впиши имя>

Генерирует варианты изменения режима, выбрасывает недопустимые, сравнивает
оставшиеся и предлагает лучший.

СЕЙЧАС: грубый перебор. Для каждого рычага берём несколько шагов вокруг
текущего значения, получаем набор сценариев, отсеиваем нарушающие ограничения,
ранжируем взвешенной суммой. Этого достаточно, чтобы закрыть требование ТЗ
("несколько допустимых вариантов -> отбросить недопустимые -> сравнить").
Никакой scipy.optimize на старте не нужен.

ЧТО ДЕЛАТЬ ДАЛЬШЕ:
  v0  подключить настоящий прогноз качества вместо линейной чувствительности
  v1  Парето-фронт вместо взвешенной суммы (ТЗ это приветствует)
  v2  комбинации из двух рычагов одновременно, а не по одному

ГЛАВНОЕ ПРАВИЛО: жёсткие ограничения проверяются ПОСЛЕ прогноза последствий
и никогда не участвуют в взвешивании. Нарушение = сценарий выбрасывается,
а не штрафуется.
"""

from __future__ import annotations

from oilcode import config
from oilcode.contracts import (
    Change,
    Constraint,
    OptimizationResult,
    ProcessState,
    QualityAssessment,
    ReliabilityAssessment,
    Scenario,
)

# Чувствительность серы к рычагам: мг/кг на ОДИН ПРОЦЕНТ изменения тега.
#
# Именно на процент, а не на единицу. Теги живут в разных масштабах: F2 это
# десятки тысяч, T5 — сотни. "Изменение на единицу" означает для них
# совершенно разное, и смешивать их в одной таблице нельзя.
#
# ДОПУЩЕНИЕ уровня "чтобы каркас заработал": знак взят из технологической
# логики, величина — оценочная. Заменить на настоящий прогноз агента качества,
# это первое, что здесь должно умереть.
SENSITIVITY_PER_PCT = {
    "T5": -0.80,   # горячее реактор -> глубже обессеривание
    "F2": -0.15,   # больше циркуляционного газа -> лучше обессеривание
    "F15": +0.25,  # больше квенча -> холоднее слой -> хуже обессеривание
    "T11": +0.40,  # больше сырья -> меньше время контакта -> хуже
    "T20": +0.05,
    "T33": +0.10,
    "F29": -0.05,
}

# Шаги подобраны операционно правдоподобными: оператор не двигает режим на 10%.
STEPS_PCT = (-0.02, -0.01, 0.01, 0.02)


class OptimizationAgent:
    def __init__(self, weights: dict[str, float] | None = None):
        # Веса критериев. Качество сюда НЕ входит: оно жёсткое ограничение,
        # а не слагаемое, которое можно перевесить выпуском.
        self.weights = weights or {
            "sulfur_margin": 1.0,
            "severity": 0.6,
            "throughput": 0.3,
            "energy": 0.2,
        }

    def optimize(
        self,
        state: ProcessState,
        quality: QualityAssessment,
        reliability: ReliabilityAssessment,
    ) -> OptimizationResult:
        result = OptimizationResult()

        if state.data_quality.overall == "insufficient":
            result.rejection_reason = (
                "Недостаточно данных: невозможно предсказать последствия изменения режима"
            )
            return result

        sulfur = state.quality.get(config.TARGET_METRIC)
        if sulfur is None:
            result.rejection_reason = f"Нет значения {config.TARGET_METRIC}"
            return result

        limit = config.HARD_LIMITS["sulfur_mg_kg_max"]
        limits_by_tag = {c.tag: c for c in reliability.constraints}

        for tag, meta in config.CONTROLS.items():
            # Только через tag_key: короткие имена тегов на двух установках
            # пересекаются, и обращение по T11 напрямую вернёт не тот сигнал.
            current = state.get(config.tag_key(tag, meta["unit"]))
            if current is None or current == 0:
                continue

            for step in STEPS_PCT:
                proposed = current * (1 + step)
                change = Change(tag=tag, current=round(current, 3),
                                proposed=round(proposed, 3))

                scenario = Scenario(id=f"{tag}{step:+.0%}", changes=[change])
                # Чувствительность задана на процент, поэтому и умножаем на проценты.
                predicted_sulfur = (
                    sulfur.value + SENSITIVITY_PER_PCT.get(tag, 0.0) * step * 100
                )
                scenario.predicted_quality = {
                    config.TARGET_METRIC: round(predicted_sulfur, 2)
                }
                scenario.predicted_severity = reliability.severity_index
                scenario.throughput_delta_pct = round(step * 100, 1) if tag in ("T11",) else 0.0
                scenario.energy_proxy_delta = round(step * 100, 1) if tag in ("T5", "F2") else 0.0

                self._check_hard_constraints(scenario, predicted_sulfur, limit,
                                             limits_by_tag, reliability)

                if scenario.hard_constraints_passed:
                    margin = limit - predicted_sulfur
                    scenario.score = round(
                        self.weights["sulfur_margin"] * margin
                        - self.weights["severity"] * scenario.predicted_severity * 10
                        + self.weights["throughput"] * scenario.throughput_delta_pct
                        - self.weights["energy"] * scenario.energy_proxy_delta,
                        3,
                    )
                result.scenarios.append(scenario)

        feasible = [s for s in result.scenarios if s.hard_constraints_passed]
        if not feasible:
            result.feasible = False
            result.rejection_reason = (
                "Все рассмотренные варианты нарушают жёсткие ограничения"
            )
            return result

        best = max(feasible, key=lambda s: s.score)
        result.feasible = True
        result.best_scenario_id = best.id
        return result

    @staticmethod
    def _check_hard_constraints(
        scenario: Scenario,
        predicted_sulfur: float,
        limit: float,
        limits_by_tag: dict[str, Constraint],
        reliability: ReliabilityAssessment,
    ) -> None:
        """Жёсткие проверки. Нарушил — выбывает, без права на компенсацию."""
        if predicted_sulfur > limit:
            scenario.hard_constraints_passed = False
            scenario.rejected_by.append(
                f"сера {predicted_sulfur:.2f} > {limit} мг/кг"
            )

        for change in scenario.changes:
            c = limits_by_tag.get(change.tag)
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

        if not reliability.regime_allowed:
            scenario.hard_constraints_passed = False
            scenario.rejected_by.append("агент надёжности запретил менять режим")
