"""Агент качества.

ЗОНА ОТВЕТСТВЕННОСТИ: <впиши имя>

Сейчас здесь baseline-заглушка: прогноз = последнее известное значение
(persistence). Это честная отправная точка — любая модель обязана её обыграть,
иначе она не нужна.

ЧТО ДЕЛАТЬ ДАЛЬШЕ (по возрастанию сложности):
  v0  сравнить persistence с формулами ВАК на истории, посчитать MAE
  v1  добавить лаги и скользящие средние по тегам реактора (T5, F2, F15, T11)
      и обучить градиентный бустинг на метках ЛИМС, сплит строго по времени
  v2  оценка неопределённости (квантильная регрессия) вместо фиксированного
      числа в confidence

ЛОВУШКА: качество отвечает на изменение режима с запаздыванием. Мгновенные
значения тегов как признаки работают плохо — нужны окна.
"""

from __future__ import annotations

from oilcode import config
from oilcode.contracts import (
    Prediction,
    ProcessState,
    QualityAssessment,
    SpecRisk,
)


class QualityAgent:
    def __init__(self, horizon_min: int = 60):
        self.horizon_min = horizon_min

    def fit(self, telemetry=None, lims=None) -> "QualityAgent":
        """Обучение. Пока не требуется — persistence не учится."""
        return self

    def assess(self, state: ProcessState) -> QualityAssessment:
        out = QualityAssessment(data_confidence=state.data_quality.overall)

        if state.data_quality.overall == "insufficient":
            out.notes.append("Данных недостаточно для оценки качества")
            return out

        for metric, meas in state.quality.items():
            # Доверие падает со старением анализа и зависит от источника.
            base = {"LIMS": 0.95, "PAK": 0.85, "VAK": 0.6}[meas.source]
            age_penalty = min((meas.age_hours or 0) / 48.0, 1.0) * 0.4
            confidence = round(max(base - age_penalty, 0.1), 2)

            out.predictions[metric] = Prediction(
                value=meas.value,
                confidence=confidence,
                horizon_min=self.horizon_min,
            )
            if meas.age_hours and meas.age_hours > 24:
                out.notes.append(
                    f"{metric}: последнее значение получено {meas.age_hours:.0f}ч назад "
                    f"({meas.source}) — прогноз ненадёжен"
                )

        # Главное жёсткое ограничение задачи.
        limit = config.HARD_LIMITS["sulfur_mg_kg_max"]
        sulfur = state.quality.get(config.TARGET_METRIC)
        if sulfur is not None:
            margin = limit - sulfur.value
            out.spec_risk[config.TARGET_METRIC] = SpecRisk(
                metric=config.TARGET_METRIC,
                exceeds_limit=sulfur.value > limit,
                margin=round(margin, 2),
                level="high" if margin < 1 else "medium" if margin < 3 else "low",
            )

        return out
