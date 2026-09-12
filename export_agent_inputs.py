"""Выгрузка ГОТОВЫХ данных для Агента качества и Агента надёжности.

Коллеги работают с уже подготовленными файлами — заглушки вычищены, простои
и мусор первых суток после пуска уже учтены в колонке usable_*, отдельно
чистить у себя не нужно. Позже это переедет в Postgres, признак usable_*
останется тем же — просто станет колонкой таблицы, а не файлом.

Это НЕ демонстрация решения целиком (для этого run_demo.py — там ещё
оптимизация и блендинг, чужая зона ответственности). Это конкретная задача
Агента данных с синка 13.09: отдать двум конкретным агентам конкретный набор
тегов с временными метками, разделённый на короткий и длинный горизонт.

    python export_agent_inputs.py

Кладёт в cache/:
  quality_agent_telemetry.csv         короткий горизонт (0-3ч), готов к использованию
  reliability_agent_telemetry.csv     длинный горизонт (недели/месяцы), готов к использованию
  quality_agent_lab_measurements.csv  сера/Т95/цетан по ЛИМС и ПАК, с available_at
  tag_manifest.csv                    справочно: какой тег кому и что значит
  excluded_periods.csv                справочно: что вычищено флагом usable_* и почему
"""

from __future__ import annotations

import sys

import pandas as pd

from oilcode import config
from oilcode.agents.data import DataAgent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

config.require_data_files()


def main() -> None:
    paths = DataAgent().export_agent_datasets()

    print("Готовые данные для агентов (usable_* уже учтён, отдельно не чистить):")
    for name in ("quality_telemetry", "reliability_telemetry", "quality_lab_measurements"):
        print(f"  {name:24} {paths[name]}")

    print("\nСправочно:")
    for name in ("tag_manifest", "excluded_periods"):
        print(f"  {name:24} {paths[name]}")

    manifest = pd.read_csv(paths["tag_manifest"])
    print(f"\nТегов агенту качества:     {(manifest['consumer'].str.contains('quality')).sum()}")
    print(f"Тегов агенту надёжности:   {(manifest['consumer'].str.contains('reliability')).sum()}")

    excluded = pd.read_csv(paths["excluded_periods"])
    real = excluded[excluded["kind"] == "outage"]
    print(f"Реальных простоев найдено: {len(real)} "
         f"(от {config.MIN_OUTAGE_HOURS:.0f}ч, шум короче — не в счёт)")


if __name__ == "__main__":
    main()
