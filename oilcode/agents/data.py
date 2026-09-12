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
from pathlib import Path

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
    """Собирает ProcessState на момент времени и выносит вердикт пригодности.

    Два разных режима работы — не путать:

    build_state(ts)           — ОДИН снимок на момент времени. Для рантайма
                                 (оркестратор дёргает это на каждое решение).
    export_agent_datasets()   — ВСЯ история по декларированным тегам, одним
                                 файлом на потребителя. Для обучения и
                                 проверки моделей — этим пользуются Агент
                                 качества и Агент надёжности у себя в коде,
                                 а не build_state().

    Решение с синка 13.09: качество живёт на коротком горизонте (0-3ч),
    надёжность — на длинном (недели/месяцы дрейфа катализатора). Поэтому
    экспорт разделён на два файла, а не один общий.
    """

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

    # ==================================================================
    # ВЫГРУЗКА ПОЛНОЙ ИСТОРИИ — то, чем пользуются агенты качества
    # и надёжности напрямую, у себя в коде, для обучения моделей.
    # ==================================================================

    def export_agent_datasets(self, out_dir: Path | None = None) -> dict[str, Path]:
        """Выгрузить три файла: короткий горизонт, длинный горизонт, лаборатория.

        Возвращает {имя: путь}. Каждый файл — временной ряд с меткой времени
        в индексе, заглушки 307/251 уже заменены на пусто, добавлен флаг
        "установка работает" на каждый момент.
        """
        out_dir = out_dir or config.CACHE
        out_dir.mkdir(exist_ok=True)

        telemetry = loaders.load_telemetry().set_index("timestamp")
        telemetry = telemetry.replace(list(config.SENTINEL_VALUES), pd.NA)
        running = self._running_flags_series(telemetry)

        paths = {
            "quality_telemetry": self._export_tag_slice(
                telemetry, running, config.QUALITY_AGENT_TAGS,
                out_dir / "quality_agent_telemetry.csv"),
            "reliability_telemetry": self._export_tag_slice(
                telemetry, running, config.RELIABILITY_AGENT_TAGS,
                out_dir / "reliability_agent_telemetry.csv"),
            "quality_lab_measurements": self._export_lab_measurements(
                out_dir / "quality_agent_lab_measurements.csv"),
        }
        return paths

    @staticmethod
    def _running_flags_series(telemetry: pd.DataFrame) -> pd.DataFrame:
        """То же правило остановки, что в _check_units_running, но на весь ряд

        разом, а не на одну строку — быстрее и даёт готовый флаг на экспорт.
        """
        medians = _flow_medians()
        flags = pd.DataFrame(index=telemetry.index)
        for unit, tags in config.UNIT_FLOW_TAGS.items():
            keys = [config.tag_key(t, unit) for t in tags if config.tag_key(t, unit) in telemetry.columns]
            if not keys:
                continue
            total = telemetry[keys].abs().sum(axis=1, skipna=True)
            median_total = sum(abs(medians.get(k, 0.0)) for k in keys)
            threshold = config.OUTAGE_FLOW_FRACTION * median_total
            flags[f"is_running_{unit}"] = (total >= threshold) if median_total > 0 else True
        return flags

    @staticmethod
    def _export_tag_slice(telemetry: pd.DataFrame, running: pd.DataFrame,
                          tag_specs: list[tuple[str, str, str]], path: Path) -> Path:
        """Срез телеметрии ровно по декларированным тегам + флаги простоя.

        Заголовок колонки = tag_key (например HT_T5) — расшифровка физического
        смысла каждого тега лежит в config.py рядом с tag_specs, здесь её
        сознательно нет, чтобы файл был чистым числовым рядом для обучения.
        """
        keys = [config.tag_key(t, u) for t, u, _ in tag_specs]
        keys = [k for k in keys if k in telemetry.columns]
        out = telemetry[keys].join(running)
        out.to_csv(path, encoding="utf-8-sig")
        return path

    @staticmethod
    def _export_lab_measurements(path: Path) -> Path:
        """Лабораторные и поточные показатели качества — длинный формат:

        timestamp | available_at | metric | value | source

        available_at учитывает 4-часовую задержку публикации ЛИМС: до этого
        момента значение не могло быть известно оператору. При обучении
        моделей использовать available_at, а не timestamp, — иначе утечка
        из будущего (то, за что ТЗ штрафует явно).
        """
        lims = loaders.load_lims()
        pak = loaders.load_pak()
        rows: list[pd.DataFrame] = []

        def add_lims(group_key: str, metrics: set[str], prefix: str = "") -> None:
            group = config.LIMS_POINTS[group_key]
            sub = lims[(lims["group"] == group) & (lims["metric"].isin(metrics))].copy()
            if sub.empty:
                return
            sub["metric"] = prefix + sub["metric"]
            sub["source"] = "LIMS"
            sub["available_at"] = sub["timestamp"] + pd.Timedelta(
                hours=config.LIMS_PUBLICATION_DELAY_H)
            rows.append(sub[["timestamp", "available_at", "metric", "value", "source"]])

        # Товарный продукт — то, что нормируется ТЗ.
        add_lims("HT_PRODUCT", {"Mg.Sulfur", "95%.T", "CetaneNumber"})
        # Сырьё на входе гидроочистки — нужно агенту качества, чтобы отделить
        # "сера скачет из-за сырья" от "сера скачет из-за режима реактора".
        add_lims("HT_FEED", {"Mass.Sulfur", "95%.T"}, prefix="feed_")

        for short, pak_tag in config.PAK_TAGS.items():
            coverage = config.SIGNAL_COVERAGE.get(f"PAK:{pak_tag}")
            sub = pak[pak["metric"] == pak_tag].copy()
            if coverage and coverage["start"]:
                sub = sub[sub["timestamp"] >= pd.Timestamp(coverage["start"])]
            if sub.empty:
                continue
            sub["metric"] = config.TARGET_METRIC if short == "sulfur" else "D15"
            sub["source"] = "PAK"
            sub["available_at"] = sub["timestamp"]  # ПАК без задержки публикации
            rows.append(sub[["timestamp", "available_at", "metric", "value", "source"]])

        out = pd.concat(rows, ignore_index=True).sort_values("timestamp")
        out.to_csv(path, index=False, encoding="utf-8-sig")
        return path
