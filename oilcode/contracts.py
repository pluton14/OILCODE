"""Контракты обмена между агентами.

Это граница между зонами ответственности пяти человек. Менять — только всей
командой, потому что от формата зависит чужой код.

Правило: агент получает на вход ProcessState (и, если нужно, выходы других
агентов) и возвращает свой dataclass. Агент НЕ читает csv/xlsx напрямую.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Confidence = Literal["ok", "degraded", "insufficient"]
Severity = Literal["normal", "elevated", "severe"]


# --------------------------------------------------------------------------
# Состояние процесса — общий вход для всех агентов
# --------------------------------------------------------------------------


@dataclass
class Measurement:
    """Одно значение показателя качества вместе с происхождением.

    Источник и возраст тащим всегда: без них нельзя оценить доверие,
    а ТЗ прямо требует учитывать свежесть анализа.
    """

    value: float
    source: Literal["LIMS", "PAK", "VAK"]
    measured_at: datetime | None
    age_hours: float | None

    def is_fresh(self, max_age_hours: float) -> bool:
        return self.age_hours is not None and self.age_hours <= max_age_hours


@dataclass
class DataQuality:
    missing_tags: list[str] = field(default_factory=list)
    stale_measurements: list[str] = field(default_factory=list)
    overall: Confidence = "ok"
    notes: list[str] = field(default_factory=list)


@dataclass
class ProcessState:
    """Снимок процесса на момент времени. Единственный вход для агентов."""

    timestamp: datetime
    kip: dict[str, float] = field(default_factory=dict)
    quality: dict[str, Measurement] = field(default_factory=dict)
    data_quality: DataQuality = field(default_factory=DataQuality)

    def get(self, tag: str, default: float | None = None) -> float | None:
        return self.kip.get(tag, default)


# --------------------------------------------------------------------------
# Выходы агентов
# --------------------------------------------------------------------------


@dataclass
class Prediction:
    value: float
    confidence: float
    horizon_min: int = 0


@dataclass
class SpecRisk:
    metric: str
    exceeds_limit: bool
    margin: float
    level: Literal["low", "medium", "high"]


@dataclass
class QualityAssessment:
    """Выход агента качества."""

    predictions: dict[str, Prediction] = field(default_factory=dict)
    spec_risk: dict[str, SpecRisk] = field(default_factory=dict)
    data_confidence: Confidence = "ok"
    notes: list[str] = field(default_factory=list)


@dataclass
class RiskFactor:
    tag: str
    issue: str
    severity: Severity


@dataclass
class Constraint:
    """Ограничение, которое агент оптимизации нарушать не имеет права."""

    tag: str
    min: float | None = None
    max: float | None = None
    reason: str = ""
    is_assumption: bool = True  # честно помечаем, что предел не из материалов


@dataclass
class ReliabilityAssessment:
    """Выход агента надёжности."""

    severity_index: float = 0.0  # 0..1
    severity_class: Severity = "normal"
    risk_factors: list[RiskFactor] = field(default_factory=list)
    regime_allowed: bool = True
    constraints: list[Constraint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class Change:
    """Одно предлагаемое изменение режима."""

    tag: str
    current: float
    proposed: float

    @property
    def delta(self) -> float:
        return self.proposed - self.current


@dataclass
class Scenario:
    id: str
    changes: list[Change] = field(default_factory=list)
    predicted_quality: dict[str, float] = field(default_factory=dict)
    predicted_severity: float = 0.0
    throughput_delta_pct: float = 0.0
    energy_proxy_delta: float = 0.0
    hard_constraints_passed: bool = True
    rejected_by: list[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class OptimizationResult:
    """Выход агента оптимизации."""

    feasible: bool = False
    scenarios: list[Scenario] = field(default_factory=list)
    best_scenario_id: str | None = None
    rejection_reason: str | None = None

    @property
    def best(self) -> Scenario | None:
        if self.best_scenario_id is None:
            return None
        return next((s for s in self.scenarios if s.id == self.best_scenario_id), None)


# --------------------------------------------------------------------------
# Финальный выход системы
# --------------------------------------------------------------------------


@dataclass
class Recommendation:
    """То, что видит оператор. Семь блоков из ТЗ.

    status = "refusal" — это НЕ ошибка, а валидный и требуемый ТЗ ответ,
    когда данных не хватает или все варианты нарушают ограничения.
    """

    timestamp: datetime
    status: Literal["recommendation", "no_action", "refusal"]

    state_summary: dict[str, str] = field(default_factory=dict)
    problem: str = ""
    action: list[Change] = field(default_factory=list)
    expected_effect: dict[str, str] = field(default_factory=dict)
    constraints_checked: list[str] = field(default_factory=list)
    confidence: str = ""
    explanation: str = ""
    alternatives: list[Scenario] = field(default_factory=list)
