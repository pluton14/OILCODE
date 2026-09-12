"""Тесты Агента данных — контракт с Агентом качества и Агентом надёжности.

Если один из этих тестов упал — значит, что-то из декларированного в
config.QUALITY_AGENT_TAGS / config.RELIABILITY_AGENT_TAGS перестало
выполняться, и агентам качества/надёжности нельзя доверять ProcessState.
"""

import pandas as pd

from oilcode import config
from oilcode.agents.data import DataAgent


def test_kip_contains_only_declared_tags():
    """kip — не дамп всей телеметрии, а ровно декларированный контракт."""
    agent = DataAgent()
    state = agent.build_state("2024-06-15 12:00")

    declared = {
        config.tag_key(tag, unit)
        for tag, unit, _ in config.QUALITY_AGENT_TAGS + config.RELIABILITY_AGENT_TAGS
    }
    assert set(state.kip.keys()) <= declared, (
        "в kip попал тег, не задекларированный ни для качества, ни для "
        "надёжности — контракт разъехался с кодом"
    )


def test_outage_is_detected_on_both_units():
    """Три известные остановки комплекса должны ловиться агентом."""
    agent = DataAgent()
    known_outages = ["2024-04-01 12:00", "2026-06-25 12:00"]
    for ts in known_outages:
        state = agent.build_state(ts)
        assert state.data_quality.overall == "insufficient", (
            f"остановка {ts} не поймана — обе установки должны быть False"
        )
        assert not any(state.data_quality.unit_running.values())


def test_sentinel_values_are_filtered_not_used_as_measurements():
    """307.00 и 251.00 — служебные заглушки, не измерения.

    Момент 2024-04-01 12:00 — внутри остановки, где заглушки заведомо есть
    (проверено вручную). Ни одно значение в kip не должно совпадать с ними.
    """
    agent = DataAgent()
    state = agent.build_state("2024-04-01 12:00")
    assert state.data_quality.sentinels_found, (
        "ожидали поймать хотя бы одну заглушку на известном простое"
    )
    for value in state.kip.values():
        assert not any(abs(value - s) < 1e-6 for s in config.SENTINEL_VALUES), (
            f"значение {value} — это заглушка, а не измерение, "
            "оно не должно было попасть в kip"
        )


def test_pak_density_absent_before_coverage_start():
    """24-2000:D15 в ПАК физически не существует раньше 2025-03-05.

    До этой даты D15 всё равно может быть известен — но только из ЛИМС
    (лаборатория меряет плотность и раньше), а не из ПАК. Агент обязан
    предупредить, что источник ПАК недоступен, а не молча взять ЛИМС.
    """
    agent = DataAgent()
    state = agent.build_state("2024-01-01 00:00")
    d15 = state.quality.get("D15")
    if d15 is not None:
        assert d15.source != "PAK", (
            "ПАК по плотности не существует до 2025-03-05, "
            "значение не могло прийти оттуда"
        )
    assert any("плотности по ПАК" in n for n in state.data_quality.notes)


def test_data_quality_is_ok_on_a_calm_moment():
    """На спокойном моменте с работающими установками отказа быть не должно."""
    agent = DataAgent()
    state = agent.build_state("2024-06-15 12:00")
    assert state.data_quality.overall in ("ok", "degraded")
    assert all(state.data_quality.unit_running.values())


def test_export_agent_datasets_is_exactly_two_files(tmp_path):
    """Ровно два файла — вход агента качества и вход агента надёжности,

    не более. Это явное требование: коллеги работают с готовыми данными,
    а не собирают их из нескольких сопроводительных файлов.
    """
    paths = DataAgent().export_agent_datasets(out_dir=tmp_path)
    assert set(paths.keys()) == {"quality", "reliability"}
    assert set(p.name for p in tmp_path.iterdir()) == {
        "quality_agent_data.csv", "reliability_agent_data.csv",
    }


def test_export_header_documents_source_and_excluded_dates(tmp_path):
    """Источник и исключённые даты — в шапке самого файла (#...), а не в

    отдельном справочнике, который можно забыть открыть.
    """
    paths = DataAgent().export_agent_datasets(out_dir=tmp_path)
    header = paths["quality"].read_text(encoding="utf-8-sig").splitlines()
    header = [ln for ln in header if ln.startswith("#")]

    assert any("HT_T5" in ln and "24-2000" in ln for ln in header), (
        "источник тега (какая установка) должен быть виден в шапке"
    )
    assert any("Периоды уже занулены" in ln for ln in header)
    assert any("24-2000:" in ln for ln in header), (
        "в quality-файле есть теги гидроочистки — её простои обязаны быть в шапке"
    )


def test_export_agent_datasets_has_no_sentinels_and_masks_outages(tmp_path):
    """То, что реально получают агенты качества/надёжности от экспорта.

    Не одна точка (build_state), а вся история — файлы, которые эти агенты
    читают у себя в коде для обучения моделей. pandas.read_csv сам пропускает
    строки шапки благодаря comment="#".
    """
    paths = DataAgent().export_agent_datasets(out_dir=tmp_path)

    quality = pd.read_csv(paths["quality"], comment="#", index_col=0, parse_dates=True)
    reliability = pd.read_csv(paths["reliability"], comment="#", index_col=0, parse_dates=True)

    for df in (quality, reliability):
        numeric = df.select_dtypes("number")
        for sentinel in config.SENTINEL_VALUES:
            assert not (numeric == sentinel).any().any(), (
                f"заглушка {sentinel} просочилась в экспорт — "
                "агент качества/надёжности примет её за измерение"
            )

    # Известная остановка АВТ (март-апрель 2024) должна быть занулена
    # во всех тегах АВТ, а не просто помечена отдельным флагом.
    during_outage = quality.loc["2024-04-01", "T33"]
    assert during_outage.isna().all(), (
        "теги АВТ на известной остановке должны быть NaN, а не значение простоя"
    )

    # Целевой показатель качества — сера, слитая из ЛИМС/ПАК прямо в тот же файл.
    assert "sulfur_mg_kg" in quality.columns
    assert "sulfur_mg_kg_source" in quality.columns
    assert quality["sulfur_mg_kg_source"].dropna().isin(["LIMS", "PAK"]).all()


def test_cetane_number_freshness_uses_its_own_sampling_interval():
    """CetaneNumber меряют раз в ~29 суток — общий 48-часовой порог здесь неверен.

    Сразу после свежего замера показатель не должен считаться устаревшим.
    """
    from oilcode.data import loaders

    lims = loaders.load_lims()
    prod = lims[(lims["group"] == config.LIMS_POINTS["HT_PRODUCT"])
               & (lims["metric"] == "CetaneNumber")].sort_values("timestamp")
    assert len(prod) > 1, "ожидали хотя бы два замера CetaneNumber"

    fresh_ts = prod.iloc[-1]["timestamp"] + pd.Timedelta(hours=5)
    agent = DataAgent()
    state = agent.build_state(fresh_ts)
    stale = state.data_quality.stale_measurements
    assert not any("CetaneNumber" in s for s in stale), (
        "показатель, измеренный 5 часов назад, не может считаться устаревшим"
    )
