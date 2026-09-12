"""Выгрузка ГОТОВЫХ данных для Агента качества и Агента надёжности.

Ровно два файла, не более. Источник и период каждого тега — в шапке самого
файла (строки с #), не в отдельном справочнике.

    python export_agent_inputs.py

Кладёт в cache/:
  quality_agent_data.csv        вход Агента качества (режим реактора + сера/Т95)
  reliability_agent_data.csv    вход Агента надёжности (износ катализатора)
"""

from __future__ import annotations

import sys

from oilcode import config
from oilcode.agents.data import DataAgent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

config.require_data_files()


def main() -> None:
    paths = DataAgent().export_agent_datasets()
    for name, path in paths.items():
        print(f"{name:12} {path}")


if __name__ == "__main__":
    main()
