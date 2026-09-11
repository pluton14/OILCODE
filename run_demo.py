"""Точка входа: один цикл принятия решения на реальных данных.

    python run_demo.py                          # момент по умолчанию
    python run_demo.py --timestamp "2024-06-15 12:00"
    python run_demo.py --scenario stale          # ситуация с устаревшими данными

Первый запуск разбирает xlsx и кладёт результат в cache/ — это занимает
минуту. Последующие запуски быстрые.
"""

from __future__ import annotations

import argparse
import sys

from oilcode.orchestrator import Orchestrator
from oilcode.report import render

# Консоль Windows по умолчанию в cp1251 и давится на юникоде.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Демонстрационные моменты времени. Подобрать осмысленно — задача команды:
# нужен спокойный период, период с риском и период с плохими данными.
SCENARIOS = {
    "normal": "2024-06-15 12:00",
    "stale": "2023-03-10 04:00",
    "recent": "2026-01-20 09:00",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="МАС: один цикл решения")
    parser.add_argument("--timestamp", help="момент времени, напр. '2024-06-15 12:00'")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="normal")
    args = parser.parse_args()

    timestamp = args.timestamp or SCENARIOS[args.scenario]

    print(f"\nЗапрошенный момент: {timestamp}")
    print("Собираю состояние процесса и опрашиваю агентов...\n")

    orchestrator = Orchestrator().fit()
    recommendation = orchestrator.decide(timestamp)
    print(render(recommendation))


if __name__ == "__main__":
    main()
