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

# Демонстрационные моменты времени. ТЗ требует показать минимум спокойный
# период, период с риском и период с неполными/аномальными данными.
SCENARIOS = {
    # Штатная работа.
    "normal": "2024-06-15 12:00",
    # Реальный инцидент: в этот день лаборатория показала 2120 мг/кг серы,
    # и примерно тогда же резко падает температура реактора — похоже на
    # замену или регенерацию катализатора. Подробности в PROCESS.md, п. 6a.
    "upset": "2024-04-23 08:00",
    # Второй инцидент поменьше: 120 и 45.9 мг/кг в один день.
    "upset2": "2025-07-24 14:00",
    # Период до начала покрытия ПАК по плотности (D15 есть только с 05.03.2025).
    "no_d15": "2024-11-05 10:00",
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
