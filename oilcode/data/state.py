"""Сборка ProcessState — снимка процесса на момент времени.

Это единственное место, где код читает исходные файлы. Агенты получают уже
готовый снимок и про существование csv/xlsx не знают.

Главное правило синхронизации: только по времени и только назад. Берём
последнее значение, измеренное НЕ ПОЗЖЕ запрошенного момента, — иначе
получим утечку информации из будущего, что ТЗ прямо запрещает.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from oilcode import config
from oilcode.contracts import DataQuality, Measurement, ProcessState
from oilcode.data import loaders


def _last_before(df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    """Последняя запись не позже ts. Ничего из будущего."""
    past = df[df["timestamp"] <= ts]
    if past.empty:
        return None
    return past.iloc[-1]


def build_process_state(timestamp: str | datetime) -> ProcessState:
    ts = pd.Timestamp(timestamp)

    telemetry = loaders.load_telemetry()
    lims = loaders.load_lims()
    pak = loaders.load_pak()

    dq = DataQuality()

    # ---- Телеметрия -------------------------------------------------------
    row = _last_before(telemetry, ts)
    kip: dict[str, float] = {}
    if row is None:
        dq.overall = "insufficient"
        dq.notes.append(f"Нет телеметрии на момент {ts}")
    else:
        for tag, value in row.items():
            if tag == "timestamp" or pd.isna(value):
                continue
            kip[str(tag)] = float(value)

        lag_min = (ts - row["timestamp"]).total_seconds() / 60
        if lag_min > 30:
            dq.overall = "degraded"
            dq.notes.append(f"Телеметрия отстаёт на {lag_min:.0f} мин")

    # Проверяем, что нужные нам рычаги и сигналы риска вообще присутствуют.
    for registry in (config.CONTROLS, config.RELIABILITY_SIGNALS):
        for tag, meta in registry.items():
            if config.tag_key(tag, meta["unit"]) not in kip:
                dq.missing_tags.append(f"{tag} ({meta['unit']})")

    # ---- Качество: ЛИМС -> ПАК -------------------------------------------
    quality: dict[str, Measurement] = {}

    target_group = config.LIMS_POINTS[config.TARGET_POINT]
    for metric in sorted(lims[lims["group"] == target_group]["metric"].unique()):
        subset = lims[(lims["group"] == target_group) & (lims["metric"] == metric)]
        last = _last_before(subset, ts)
        if last is None:
            continue
        age_h = (ts - last["timestamp"]).total_seconds() / 3600
        quality[metric] = Measurement(
            value=float(last["value"]),
            source="LIMS",
            measured_at=last["timestamp"].to_pydatetime(),
            age_hours=age_h,
        )
        if age_h > config.FRESHNESS["lims_max_age_h"]:
            dq.stale_measurements.append(f"LIMS:{metric} ({age_h:.0f}ч)")

    # ПАК заменяет ЛИМС там, где лабораторный результат устарел или его нет.
    for short, pak_tag in config.PAK_TAGS.items():
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
                value=float(last["value"]),
                source="PAK",
                measured_at=last["timestamp"].to_pydatetime(),
                age_hours=age_h,
            )
        if age_h * 60 > config.FRESHNESS["pak_max_age_min"]:
            dq.stale_measurements.append(f"PAK:{pak_tag} ({age_h:.1f}ч)")

    # ---- Итоговая оценка пригодности данных -------------------------------
    if config.TARGET_METRIC not in quality:
        dq.overall = "insufficient"
        dq.notes.append(
            f"Нет ни одного источника по {config.TARGET_METRIC} — "
            "решение принимать не на чем"
        )
    elif dq.stale_measurements and dq.overall == "ok":
        dq.overall = "degraded"

    return ProcessState(timestamp=ts.to_pydatetime(), kip=kip,
                        quality=quality, data_quality=dq)
