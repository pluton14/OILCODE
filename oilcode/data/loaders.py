"""Чтение исходных файлов.

Файлы лежат в неудобном виде: ЛИМС и ПАК — это пары колонок (дата, значение)
на каждый показатель, с разным покрытием по времени. Здесь они приводятся
к длинному формату: point | metric | timestamp | value.

ПАК-файл большой (~190 тыс. строк), поэтому распарсенные таблицы кэшируются
в cache/*.csv. Удалить кэш — просто стереть папку.
"""

from __future__ import annotations

import pandas as pd

from oilcode import config


# --------------------------------------------------------------------------
# Вспомогательное
# --------------------------------------------------------------------------


def _cached(name: str, build):
    """Построить таблицу один раз и переиспользовать из cache/."""
    config.CACHE.mkdir(exist_ok=True)
    path = config.CACHE / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["timestamp"])
    df = build()
    df.to_csv(path, index=False)
    return df


def _pairs_to_long(raw: pd.DataFrame, header_row: int, first_data_row: int,
                   extra_header_row: int | None = None) -> pd.DataFrame:
    """Развернуть таблицу из пар колонок (дата, значение) в длинный формат.

    В строке header_row стоит имя показателя — но только над колонкой с датой.
    Колонка справа от неё содержит значения.
    """
    names = raw.iloc[header_row]
    groups = raw.iloc[extra_header_row].ffill() if extra_header_row is not None else None

    records = []
    for col in range(raw.shape[1] - 1):
        metric = names.iloc[col]
        if pd.isna(metric):
            continue
        ts = pd.to_datetime(raw.iloc[first_data_row:, col], errors="coerce")
        val = pd.to_numeric(raw.iloc[first_data_row:, col + 1], errors="coerce")
        ok = ts.notna() & val.notna()
        if not ok.any():
            continue
        chunk = pd.DataFrame({"timestamp": ts[ok], "value": val[ok]})
        chunk["metric"] = str(metric).strip()
        chunk["group"] = str(groups.iloc[col]).strip() if groups is not None else ""
        records.append(chunk)

    if not records:
        return pd.DataFrame(columns=["group", "metric", "timestamp", "value"])
    out = pd.concat(records, ignore_index=True)
    return out[["group", "metric", "timestamp", "value"]].sort_values("timestamp")


# --------------------------------------------------------------------------
# Телеметрия
# --------------------------------------------------------------------------


def load_telemetry() -> pd.DataFrame:
    """Телеметрия обеих установок, склеенная по времени.

    Шаг 10 минут. Служебные колонки Unnamed выбрасываются.
    Теги АВТ и 24-2000 частично совпадают по имени (T6, F9 и др.), поэтому
    теги 24-2000 получают префикс 'HT_'.
    """

    def build() -> pd.DataFrame:
        avt = pd.read_csv(config.AVT_TAGS_CSV)
        ht = pd.read_csv(config.HT_TAGS_CSV)

        for df in (avt, ht):
            df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")],
                    inplace=True, errors="ignore")
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        ht = ht.rename(columns={c: f"HT_{c}" for c in ht.columns if c != "date"})
        merged = pd.merge(avt, ht, on="date", how="outer").sort_values("date")
        return merged.rename(columns={"date": "timestamp"})

    return _cached("telemetry", build)


# --------------------------------------------------------------------------
# ЛИМС
# --------------------------------------------------------------------------


def load_lims() -> pd.DataFrame:
    """Лабораторные анализы: group = точка отбора, metric = показатель.

    Осторожно: строка единиц измерения в исходном файле местами съехала
    (у D15 написано °С при значениях ~876 кг/м3), поэтому она не читается.
    """

    def build() -> pd.DataFrame:
        raw = pd.read_excel(config.LIMS_XLSX, header=None)
        return _pairs_to_long(raw, header_row=1, first_data_row=4, extra_header_row=0)

    return _cached("lims", build)


# --------------------------------------------------------------------------
# ПАК
# --------------------------------------------------------------------------


def load_pak() -> pd.DataFrame:
    """Поточные анализаторы. Всего два сигнала.

    ВНИМАНИЕ: 24-2000:D15 существует только с 2025-03-05. Сера — с 2023-01-01.
    """

    def build() -> pd.DataFrame:
        raw = pd.read_excel(config.PAK_XLSX, header=None)
        return _pairs_to_long(raw, header_row=0, first_data_row=2)

    return _cached("pak", build)


# --------------------------------------------------------------------------
# Справочники
# --------------------------------------------------------------------------


def load_kip_dictionary() -> pd.DataFrame:
    """Расшифровка тегов: tag | description | unit.

    Единственный законный источник смысла тега. Буквенный префикс на 24-2000
    физическому смыслу НЕ соответствует.
    """

    def build() -> pd.DataFrame:
        raw = pd.read_excel(config.TAGS_XLSX, sheet_name="КИП", header=0)
        cols = list(raw.columns)
        rows = []
        for desc_col, tag_col, unit in ((cols[0], cols[1], "AVT"),
                                        (cols[2], cols[3], "24-2000")):
            part = raw[[tag_col, desc_col]].dropna()
            part.columns = ["tag", "description"]
            part["unit"] = unit
            rows.append(part)
        return pd.concat(rows, ignore_index=True)

    path = config.CACHE / "kip.csv"
    config.CACHE.mkdir(exist_ok=True)
    if path.exists():
        return pd.read_csv(path)
    df = build()
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def load_vak_formulas() -> pd.DataFrame:
    """Формулы виртуальных анализаторов: name | formula.

    Их не нужно изобретать — они уже даны. Используем как baseline
    в агенте качества, прежде чем обучать что-то своё.
    """
    raw = pd.read_excel(config.TAGS_XLSX, sheet_name="ВАК", header=None)
    records = []
    for col in range(raw.shape[1] - 1):
        for row in range(1, raw.shape[0]):
            name, formula = raw.iloc[row, col], raw.iloc[row, col + 1]
            if isinstance(name, str) and isinstance(formula, str) and ":" in name:
                records.append({"name": name.strip(), "formula": formula.strip()})
    return pd.DataFrame(records).drop_duplicates()
