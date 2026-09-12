"""Агент данных.

ЗОНА ОТВЕТСТВЕННОСТИ: <впиши имя>

Закрывает шаги 1-2 цикла принятия решения из ТЗ:
    1. получить актуальное состояние процесса из телеметрии и анализов;
    2. проверить полноту, актуальность и согласованность данных.

Это ЕДИНСТВЕННЫЙ агент, который читает сырые csv/xlsx. Все остальные
(качество, надёжность, оптимизация, блендинг) получают только ProcessState
и понятия не имеют, что где-то на диске лежит avt_tags.csv. Если агенту
качества или надёжности понадобился новый тег — его дописывают в
config.QUALITY_AGENT_TAGS / config.RELIABILITY_AGENT_TAGS, а не читают
телеметрию в обход этого класса.

ЧТО ИМЕННО ОТДАЁТСЯ ДАЛЬШЕ (контракт зафиксирован в config.py):
  Агенту качества     -> ProcessState.kip по тегам config.QUALITY_AGENT_TAGS
                          + ProcessState.quality (сера, Т95 из ЛИМС/ПАК)
  Агенту надёжности   -> ProcessState.kip по тегам config.RELIABILITY_AGENT_TAGS
  Обоим                -> ProcessState.data_quality — и это ОБЯЗАТЕЛЬНО к
                          прочтению до того, как использовать что-либо ещё:
                          при data_quality.overall == "insufficient" оба
                          агента обязаны отказаться от оценки, а не считать
                          "на всякий случай".

ПРАВО ВЕТО: overall == "insufficient" останавливает всю цепочку на
оркестраторе, дальше агенты даже не опрашиваются (см. orchestrator.py).

ЧТО ДЕЛАТЬ ДАЛЬШЕ:
  v0  детектор аномалий поверх правил (Isolation Forest) — ТЗ относит
      к "приветствуется"
  v1  расширить SIGNAL_COVERAGE, если найдутся ещё сигналы с ограниченным
      покрытием (искать так же, как нашли PAK D15)
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache

import pandas as pd

from oilcode import config
from oilcode.contracts import DataQuality, Measurement, ProcessState
from oilcode.data import loaders


def _last_before(df: pd.DataFrame, ts: pd.Timestamp,
                 delay_h: float = 0.0) -> pd.Series | None:
    """Последняя запись, которая к моменту ts уже была ИЗВЕСТНА.

    delay_h — задержка между измерением и его появлением в системе.
    Для ЛИМС это 4 часа (метка времени = момент отбора пробы, результат
    публикуется позже). Без этой поправки мы используем лабораторный
    результат раньше, чем его мог увидеть оператор, — то есть подглядываем
    в будущее, что ТЗ прямо запрещает.
    """
    available_at = df["timestamp"] + pd.Timedelta(hours=delay_h)
    past = df[available_at <= ts]
    if past.empty:
        return None
    return past.iloc[-1]


@lru_cache(maxsize=1)
def _flow_medians() -> dict[str, float]:
    """Медиана каждого расходного тега по всей истории — точка отсчёта

    для определения "установка стоит". Считается один раз и кешируется:
    файл телеметрии большой, пересчитывать на каждый вызов незачем.
    """
    telemetry = loaders.load_telemetry()
    medians: dict[str, float] = {}
    for unit, tags in config.UNIT_FLOW_TAGS.items():
        for tag in tags:
            key = config.tag_key(tag, unit)
            if key in telemetry.columns:
                medians[key] = float(telemetry[key].median())
    return medians


class DataAgent:
    """Собирает ProcessState на момент времени и выносит вердикт пригодности."""

    def build_state(self, timestamp: str | datetime) -> ProcessState:
        ts = pd.Timestamp(timestamp)

        telemetry = loaders.load_telemetry()
        lims = loaders.load_lims()
        pak = loaders.load_pak()

        dq = DataQuality()
        row = _last_before(telemetry, ts)
        kip: dict[str, float] = {}

        if row is None:
            dq.overall = "insufficient"
            dq.notes.append(f"Нет телеметрии на момент {ts}")
        else:
            lag_min = (ts - row["timestamp"]).total_seconds() / 60
            if lag_min > 30:
                dq.overall = "degraded"
                dq.notes.append(f"Телеметрия отстаёт на {lag_min:.0f} мин")

            dq.unit_running = self._check_units_running(row)
            for unit, running in dq.unit_running.items():
                if not running:
                    dq.notes.append(
                        f"Установка {unit} не работает на этот момент "
                        "(расход ниже порога) — её теги не отражают режим"
                    )

            kip = self._extract_declared_tags(row, dq)

        # ---- Качество: ЛИМС -> ПАК ---------------------------------------
        quality = self._collect_quality(lims, pak, ts, dq)

        # ---- Итоговый вердикт ----------------------------------------------
        if config.TARGET_METRIC not in quality:
            dq.overall = "insufficient"
            dq.notes.append(
                f"Нет ни одного источника по {config.TARGET_METRIC} — "
                "решение принимать не на чем"
            )
        elif row is not None and not any(dq.unit_running.values()):
            dq.overall = "insufficient"
            dq.notes.append("Обе установки не работают — оценивать нечего")
        elif dq.overall == "ok" and (dq.stale_measurements or dq.sentinels_found
                                     or dq.missing_tags):
            dq.overall = "degraded"

        return ProcessState(timestamp=ts.to_pydatetime(), kip=kip,
                            quality=quality, data_quality=dq)

    # ------------------------------------------------------------------
    # Работает ли установка
    # ------------------------------------------------------------------

    @staticmethod
    def _check_units_running(row: pd.Series) -> dict[str, bool]:
        """Установка считается стоящей, если её расход упал ниже доли медианы.

        Проверено на трёх фактических остановках комплекса — правило ловит
        все три (см. AVT_TAGS.md, HT_TAGS.md).
        """
        medians = _flow_medians()
        result = {}
        for unit, tags in config.UNIT_FLOW_TAGS.items():
            total = 0.0
            median_total = 0.0
            for tag in tags:
                key = config.tag_key(tag, unit)
                val = row.get(key)
                median = medians.get(key, 0.0)
                if val is not None and not pd.isna(val):
                    total += abs(float(val))
                median_total += abs(median)
            threshold = config.OUTAGE_FLOW_FRACTION * median_total
            result[unit] = total >= threshold if median_total > 0 else True
        return result

    # ------------------------------------------------------------------
    # Только декларированные теги, с фильтром заглушек
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_declared_tags(row: pd.Series, dq: DataQuality) -> dict[str, float]:
        """Кладёт в kip ровно те теги, что нужны агентам качества и надёжности —

        не все 97 колонок телеметрии. Заглушки 307.00/251.00 не пропускаются:
        это не измерения, использовать их как измерение — ошибка.
        """
        kip: dict[str, float] = {}
        needed = {
            (tag, unit)
            for tag, unit, _ in config.QUALITY_AGENT_TAGS + config.RELIABILITY_AGENT_TAGS
        }
        for tag, unit in sorted(needed):
            key = config.tag_key(tag, unit)
            val = row.get(key)
            if val is None or pd.isna(val):
                dq.missing_tags.append(f"{tag} ({unit})")
                continue
            val = float(val)
            if any(abs(val - s) < 1e-6 for s in config.SENTINEL_VALUES):
                dq.sentinels_found.append(f"{tag} ({unit}) = {val:g}")
                continue
            kip[key] = val
        return kip

    # ------------------------------------------------------------------
    # Качество: приоритет ЛИМС -> ПАК, с учётом задержки и покрытия
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_quality(lims: pd.DataFrame, pak: pd.DataFrame, ts: pd.Timestamp,
                         dq: DataQuality) -> dict[str, Measurement]:
        quality: dict[str, Measurement] = {}

        target_group = config.LIMS_POINTS[config.TARGET_POINT]
        for metric in sorted(lims[lims["group"] == target_group]["metric"].unique()):
            subset = lims[(lims["group"] == target_group) & (lims["metric"] == metric)]
            last = _last_before(subset, ts, delay_h=config.LIMS_PUBLICATION_DELAY_H)
            if last is None:
                continue
            # Возраст считаем от МОМЕНТА ОТБОРА: оператору важно, насколько
            # устарела сама проба, а не когда её напечатали.
            age_h = (ts - last["timestamp"]).total_seconds() / 3600
            quality[metric] = Measurement(
                value=float(last["value"]), source="LIMS",
                measured_at=last["timestamp"].to_pydatetime(), age_hours=age_h,
            )
            # Порог свежести свой у каждого показателя: CetaneNumber меряют
            # раз в 29 суток, и общий 48-часовой порог объявлял бы его
            # устаревшим постоянно, даже сразу после свежей пробы.
            threshold = config.LIMS_METRIC_FRESHNESS_H.get(
                metric, config.FRESHNESS["lims_max_age_h"])
            if age_h > threshold:
                dq.stale_measurements.append(f"LIMS:{metric} ({age_h:.0f}ч)")

        # ПАК заменяет ЛИМС там, где лабораторный результат устарел или его нет.
        for short, pak_tag in config.PAK_TAGS.items():
            coverage = config.SIGNAL_COVERAGE.get(f"PAK:{pak_tag}")
            if coverage and coverage["start"] and ts < pd.Timestamp(coverage["start"]):
                dq.notes.append(f"ПАК {pak_tag}: {coverage['note']}")
                continue

            subset = pak[pak["metric"] == pak_tag]
            last = _last_before(subset, ts)
            if last is None:
                dq.notes.append(f"ПАК {pak_tag}: нет данных на этот момент")
                continue
            age_h = (ts - last["timestamp"]).total_seconds() / 3600

            metric_key = config.TARGET_METRIC if short == "sulfur" else "D15"
            existing = quality.get(metric_key)
            if existing is None or (existing.age_hours or 1e9) > age_h:
                quality[metric_key] = Measurement(
                    value=float(last["value"]), source="PAK",
                    measured_at=last["timestamp"].to_pydatetime(), age_hours=age_h,
                )
            if age_h * 60 > config.FRESHNESS["pak_max_age_min"]:
                dq.stale_measurements.append(f"PAK:{pak_tag} ({age_h:.1f}ч)")

        return quality
