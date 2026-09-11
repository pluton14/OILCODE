"""Агент надёжности.

ЗОНА ОТВЕТСТВЕННОСТИ: <впиши имя>

Оценивает тяжесть режима и риск для оборудования/катализатора.

ПОЧЕМУ ЭТО ВООБЩЕ НУЖНО: на гидроочистке всегда можно "догнать" качество,
подняв температуру реактора. Но чем горячее — тем быстрее садится катализатор
и растёт перепад давления на слое. Замена катализатора = остановка установки.
Именно поэтому решение нельзя принимать одним только агентом качества.

Прямой физический индикатор износа у нас есть: W10 — перепад давления на
реакторе Р-202. Рост перепада = закоксовывание. Это не прокси.

СЕЙЧАС: перцентильная оценка — насколько текущее значение сигнала высоко
относительно собственной истории.

ЧТО ДЕЛАТЬ ДАЛЬШЕ:
  v0  добавить длительность: "сколько часов подряд выше p95" важнее, чем
      мгновенное превышение
  v1  тренд перепада давления за недели (скорость закоксовывания)
  v2  детектор аномалий (IsolationForest) на сочетаниях сигналов

ДОПУЩЕНИЕ: промышленных пределов нам не выдали, поэтому "нормой" считается
поведение самого сигнала в истории. Это модельное допущение, не паспортный
предел оборудования — так и указываем оператору.
"""

from __future__ import annotations

import pandas as pd

from oilcode import config
from oilcode.contracts import (
    Constraint,
    ProcessState,
    ReliabilityAssessment,
    RiskFactor,
)


class ReliabilityAgent:
    def __init__(self):
        self.quantiles: dict[str, dict[str, float]] = {}

    def fit(self, telemetry: pd.DataFrame) -> "ReliabilityAgent":
        """Калибровка на истории: что для этого сигнала вообще нормально."""
        for tag, meta in config.RELIABILITY_SIGNALS.items():
            col = config.tag_key(tag, meta["unit"])
            if col not in telemetry.columns:
                continue
            series = telemetry[col].dropna()
            if series.empty:
                continue
            self.quantiles[tag] = {
                "p50": float(series.quantile(0.50)),
                "p90": float(series.quantile(0.90)),
                "p95": float(series.quantile(0.95)),
                "p99": float(series.quantile(0.99)),
            }
        return self

    def assess(self, state: ProcessState) -> ReliabilityAssessment:
        out = ReliabilityAssessment()
        scores: list[float] = []

        for tag, meta in config.RELIABILITY_SIGNALS.items():
            value = state.get(config.tag_key(tag, meta["unit"]))
            q = self.quantiles.get(tag)
            if value is None or q is None:
                out.notes.append(f"{tag}: нет данных или калибровки — сигнал пропущен")
                continue

            if value >= q["p99"]:
                score, severity = 1.0, "severe"
            elif value >= q["p95"]:
                score, severity = 0.75, "elevated"
            elif value >= q["p90"]:
                score, severity = 0.5, "elevated"
            else:
                score, severity = max((value - q["p50"]) / (q["p90"] - q["p50"] + 1e-9), 0) * 0.4, "normal"

            scores.append(min(score, 1.0))

            if severity != "normal":
                out.risk_factors.append(RiskFactor(
                    tag=tag,
                    issue=f"{meta['desc']}: {value:.2f} — выше p{95 if score >= 0.75 else 90} "
                          f"({q['p95' if score >= 0.75 else 'p90']:.2f}). {meta['meaning']}",
                    severity=severity,
                ))

            # Ограничение для агента оптимизации: не разгонять сигнал дальше p95.
            control_tags = {m["tag"] for m in config.CONTROLS.values()}
            if tag in control_tags:
                out.constraints.append(Constraint(
                    tag=tag,
                    max=q["p95"],
                    reason=f"модельный предел по истории (p95); {meta['meaning']}",
                    is_assumption=True,
                ))

        out.severity_index = round(max(scores) if scores else 0.0, 2)
        out.severity_class = (
            "severe" if out.severity_index >= 0.95
            else "elevated" if out.severity_index >= 0.5
            else "normal"
        )
        out.regime_allowed = out.severity_class != "severe"
        if not out.regime_allowed:
            out.notes.append("Режим за пределами исторически наблюдавшегося — "
                             "наращивать нагрузку нельзя")
        return out
