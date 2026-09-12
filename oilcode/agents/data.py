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
        """Выгрузить РОВНО ДВА файла — вход Агента качества и вход Агента

        надёжности. Ничего третьего. Заглушки 307/251 уже пусто, периоды
        простоя и первые сутки после пуска (мусор переходного режима) уже
        вычищены в NaN — отдельно чистить у себя не нужно.

        Детализация источника и дат — не в третьем файле, а прямо в шапке
        каждого из двух: несколько строк-комментариев (#), которые
        pandas.read_csv(..., comment="#") пропускает сам.
        """
        out_dir = out_dir or config.CACHE
        out_dir.mkdir(exist_ok=True)

        telemetry = loaders.load_telemetry().set_index("timestamp").sort_index()
        telemetry = telemetry.replace(list(config.SENTINEL_VALUES), pd.NA)
        running = self._running_flags_series(telemetry)
        excluded = self._excluded_periods(running)  # только для маски и шапки, не файл

        quality_path = self._export_dataset(
            telemetry, running, excluded, config.QUALITY_AGENT_TAGS,
            out_dir / "quality_agent_data.csv", agent_name="Агент качества",
            lab_metrics={"sulfur_mg_kg": "Mg.Sulfur", "t95_c": "95%.T"})

        reliability_path = self._export_dataset(
            telemetry, running, excluded, config.RELIABILITY_AGENT_TAGS,
            out_dir / "reliability_agent_data.csv", agent_name="Агент надёжности",
            lab_metrics=None)

        return {"quality": quality_path, "reliability": reliability_path}

    @staticmethod
    def _false_runs(flag: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Границы непрерывных участков, где flag == False, очищенные от шума.

        На шаге 10 минут одиночный шумовой провал расхода тоже помечается
        "не работает" — без сглаживания три настоящих остановки тонут в
        десятках ложных 10-30-минутных. Поэтому: сначала близкие провалы
        (разрыв короче MERGE_OUTAGE_GAP_HOURS) объединяются в один, потом
        всё короче MIN_OUTAGE_HOURS отбрасывается как измерительный шум.
        """
        down = ~flag.fillna(False)
        if not down.any():
            return []
        group = (down != down.shift()).cumsum()
        raw = [(idx.min(), idx.max()) for _, idx in down[down].groupby(group[down]).groups.items()]
        raw.sort()

        merged: list[list[pd.Timestamp]] = []
        gap = pd.Timedelta(hours=config.MERGE_OUTAGE_GAP_HOURS)
        for start, end in raw:
            if merged and start - merged[-1][1] <= gap:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        min_dur = pd.Timedelta(hours=config.MIN_OUTAGE_HOURS)
        return [(s, e) for s, e in merged if e - s >= min_dur]

    @classmethod
    def _excluded_periods(cls, running: pd.DataFrame) -> list[dict]:
        """Периоды на исключение — outage (простой) и startup (сутки после пуска).

        Выводится ИЗ ДАННЫХ при каждом запуске, не хранится списком дат
        руками — не может разъехаться с фактическим рядом. Используется и
        как маска (какие ячейки занулить), и как текст в шапке файла.
        """
        rows = []
        for unit in config.UNIT_FLOW_TAGS:
            col = f"is_running_{unit}"
            if col not in running.columns:
                continue
            for start, end in cls._false_runs(running[col]):
                rows.append({"unit": unit, "kind": "outage", "start": start, "end": end})
                rows.append({"unit": unit, "kind": "startup", "start": end,
                            "end": end + pd.Timedelta(hours=config.STARTUP_BUFFER_HOURS)})
        return rows

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

    @classmethod
    def _export_dataset(cls, telemetry: pd.DataFrame, running: pd.DataFrame,
                        excluded: list[dict], tag_specs: list[tuple[str, str, str]],
                        path: Path, agent_name: str,
                        lab_metrics: dict[str, str] | None) -> Path:
        """Один файл — один агент. Теги + (для качества) слитые лабораторные

        показатели, с занулёнными периодами простоя/пуска и шапкой-описанием
        источника и дат прямо в CSV (строки '#', pandas.read_csv их
        игнорирует сам).
        """
        keys = [config.tag_key(t, u) for t, u, _ in tag_specs]
        keys = [k for k in keys if k in telemetry.columns]
        out = telemetry[keys].copy()

        # Занулить периоды простоя/пуска — только те юниты, к которым тег
        # относится (тег АВТ не зависит от того, стоит ли 24-2000, и наоборот).
        for tag, unit, _ in tag_specs:
            key = config.tag_key(tag, unit)
            if key not in out.columns:
                continue
            col = f"is_running_{unit}"
            if col not in running.columns:
                continue
            bad = ~running[col].fillna(False)
            for _, end in cls._false_runs(running[col]):
                bad |= (out.index > end) & (out.index <= end + pd.Timedelta(
                    hours=config.STARTUP_BUFFER_HOURS))
            out.loc[bad, key] = pd.NA

        if lab_metrics:
            for out_col, metric in lab_metrics.items():
                value, source = cls._merged_lab_metric(metric, out.index)
                out[out_col] = value
                out[f"{out_col}_source"] = source

        header = cls._dataset_header(agent_name, tag_specs, excluded, lab_metrics)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            f.write(header)
            out.to_csv(f)
        return path

    @staticmethod
    def _dataset_header(agent_name: str, tag_specs: list[tuple[str, str, str]],
                        excluded: list[dict], lab_metrics: dict[str, str] | None) -> str:
        lines = [f"# {agent_name} — вход. Источник и что значит каждая колонка:"]
        for tag, unit, desc in tag_specs:
            src = "телеметрия АВТ" if unit == "AVT" else "телеметрия 24-2000"
            lines.append(f"# {config.tag_key(tag, unit):8} {src:18} {desc}")
        if lab_metrics:
            lines.append(f"# {'sulfur_mg_kg':8} {'ЛИМС/ПАК':18} сера, целевой показатель, приоритет — самый свежий источник")
            lines.append(f"# {'t95_c':8} {'ЛИМС':18} температура 95% выкипания товарного продукта")
            lines.append("# *_source показывает, откуда взято конкретное значение: LIMS или PAK")
        lines.append("#")
        lines.append("# Периоды уже занулены (простой установки + "
                     f"{config.STARTUP_BUFFER_HOURS:.0f}ч после пуска), отдельно чистить не нужно:")
        for unit in config.UNIT_FLOW_TAGS:
            outages = [e for e in excluded if e["unit"] == unit and e["kind"] == "outage"]
            if not outages:
                continue
            span = ", ".join(f"{e['start']:%Y-%m-%d}..{e['end']:%Y-%m-%d}" for e in outages)
            lines.append(f"#   {unit}: {span}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _merged_lab_metric(metric: str, index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
        """Значение метрики на каждый момент телеметрии — самый свежий из

        известных на этот момент источников (ЛИМС и/или ПАК), с учётом
        4-часовой задержки публикации ЛИМС. Именно так же расставлен
        приоритет в build_state() — не должно разъезжаться с рантаймом.
        """
        lims = loaders.load_lims()
        pak = loaders.load_pak()
        rows: list[pd.DataFrame] = []

        product = lims[(lims["group"] == config.LIMS_POINTS["HT_PRODUCT"])
                       & (lims["metric"] == metric)].copy()
        if not product.empty:
            product["available_at"] = product["timestamp"] + pd.Timedelta(
                hours=config.LIMS_PUBLICATION_DELAY_H)
            product["source"] = "LIMS"
            rows.append(product[["available_at", "value", "source"]])

        pak_tag = {"Mg.Sulfur": config.PAK_TAGS.get("sulfur")}.get(metric)
        if pak_tag:
            coverage = config.SIGNAL_COVERAGE.get(f"PAK:{pak_tag}")
            sub = pak[pak["metric"] == pak_tag].copy()
            if coverage and coverage["start"]:
                sub = sub[sub["timestamp"] >= pd.Timestamp(coverage["start"])]
            if not sub.empty:
                sub["available_at"] = sub["timestamp"]  # ПАК без задержки публикации
                sub["source"] = "PAK"
                rows.append(sub[["available_at", "value", "source"]])

        if not rows:
            empty = pd.Series(pd.NA, index=index)
            return empty, empty.astype(object)

        combined = pd.concat(rows, ignore_index=True).sort_values("available_at")
        merged = pd.merge_asof(
            pd.DataFrame({"timestamp": index}), combined,
            left_on="timestamp", right_on="available_at", direction="backward",
        )
        return merged["value"].to_numpy(), merged["source"].to_numpy()

