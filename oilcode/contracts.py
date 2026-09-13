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

import pandas as pd

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
    """Вердикт Агента данных. Это то, на что смотрят все остальные агенты,

    прежде чем доверять ProcessState.kip и ProcessState.quality.
    """

    missing_tags: list[str] = field(default_factory=list)
    stale_measurements: list[str] = field(default_factory=list)
    # Работает ли каждая установка на этот момент. {"AVT": True, "24-2000": False}
    # Если установка стоит — её теги физически не значат то же самое (расход
    # в нуле, температура падает к уличной), это не аномалия режима.
    unit_running: dict[str, bool] = field(default_factory=dict)
    # Теги, которые в этот момент показывают служебную заглушку (307/251),
    # а не измерение. Такой тег не попал в kip — вместо него запись здесь.
    sentinels_found: list[str] = field(default_factory=list)
    overall: Confidence = "ok"
    notes: list[str] = field(default_factory=list)


@dataclass
class ProcessState:
    """Снимок процесса на момент времени. Единственный вход для агентов.

    ПОЧЕМУ ЗДЕСЬ ЕСТЬ ИСТОРИЯ, А НЕ ТОЛЬКО ТОЧКА.
    Качество отвечает на изменение режима с запаздыванием 0-3 часа, поэтому
    по одному мгновенному срезу тегов прогноз не строится в принципе — нужны
    окна и лаги. Раньше каждый агент решал бы это сам, лазая в телеметрию в
    обход Агента данных; теперь окно приезжает готовым, уже очищенным от
    заглушек и простоев, и правило "никто кроме Агента данных не читает
    сырые файлы" остаётся в силе.

    history — DataFrame с DatetimeIndex по тегам (колонки — ключи как в kip),
    строго ДО timestamp включительно. Ничего из будущего в нём нет.
    """

    timestamp: datetime
    kip: dict[str, float] = field(default_factory=dict)
    quality: dict[str, Measurement] = field(default_factory=dict)
    data_quality: DataQuality = field(default_factory=DataQuality)
    history: "pd.DataFrame | None" = None
    # История целевого показателя (сера) — отдельно, потому что живёт в
    # другом темпе: ПАК раз в 10 минут, ЛИМС раз в сутки.
    target_history: "pd.Series | None" = None

    def get(self, tag: str, default: float | None = None) -> float | None:
        return self.kip.get(tag, default)


# --------------------------------------------------------------------------
# Выходы агентов
# --------------------------------------------------------------------------


@dataclass
class Prediction:
    """Одно предсказание показателя на один горизонт.

    lo/hi — интервал, а не украшение: агент оптимизации обязан считать риск
    по верхней границе, иначе рекомендация "запас 0.3 мг/кг" выглядит
    безопасной, хотя разброс модели ±2.
    basis — на чём построено, чтобы в логе решений было видно, сработала
    режимная модель или мы упали на инерцию.
    """

    value: float
    confidence: float
    horizon_min: int = 0
    lo: float | None = None
    hi: float | None = None
    basis: str = ""


@dataclass
class SpecRisk:
    metric: str
    exceeds_limit: bool
    margin: float
    level: Literal["low", "medium", "high"]


@dataclass
class QualityAssessment:
    """Выход агента качества.

    predictions — заголовочный прогноз по каждому показателю (на основном
    горизонте). forecast — тот же прогноз, но по всем горизонтам сразу:
    на 1ч инерция процесса почти непобедима, на 3ч у режимной модели
    появляется шанс, и оператору важно видеть это раздельно.
    """

    predictions: dict[str, Prediction] = field(default_factory=dict)
    forecast: dict[str, list[Prediction]] = field(default_factory=dict)
    spec_risk: dict[str, SpecRisk] = field(default_factory=dict)
    data_confidence: Confidence = "ok"
    # Насколько модель обыграла инерцию на валидации, в процентах MAE.
    # Отрицательное значение — модель хуже инерции, и тогда она не
    # применяется вовсе (см. QualityAgent.fit).
    model_gain_pct: dict[int, float] = field(default_factory=dict)
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
class BlendRecipe:
    """Рецептура смешения товарного ДТ.

    Доли резервуаров обязаны давать 100%, присадка дозируется сверх этого
    и ограничена 3% по массе.
    """

    fractions: dict[str, float] = field(default_factory=dict)  # доли, сумма = 1.0
    improver_pct: float = 0.0
    blended: dict[str, float] = field(default_factory=dict)  # сера, Т95, цетан
    cost: float = 0.0
    meets_spec: bool = False
    violations: list[str] = field(default_factory=list)


@dataclass
class BlendingResult:
    """Выход агента блендинга."""

    feasible: bool = False
    best: BlendRecipe | None = None
    considered: int = 0
    rejection_reason: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Выход агента оптимизации."""

    feasible: bool = False
    scenarios: list[Scenario] = field(default_factory=list)
    best_scenario_id: str | None = None
    rejection_reason: str | None = None
    # Сценарий "ничего не менять" — точка отсчёта, с которой сравнивается
    # выгода от вмешательства.
    baseline_score: float | None = None

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
