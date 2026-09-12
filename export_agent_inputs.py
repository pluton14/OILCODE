"""Выгрузка исторических данных для Агента качества и Агента надёжности.

Это НЕ демонстрация решения целиком (для этого run_demo.py — там ещё
оптимизация и блендинг, чужая зона ответственности). Это конкретная задача
Агента данных с синка 13.09: отдать двум конкретным агентам конкретный набор
тегов с временными метками, разделённый на короткий и длинный горизонт.

    python export_agent_inputs.py

Кладёт в cache/:
  quality_agent_telemetry.csv         короткий горизонт (0-3ч) — режим реактора
  quality_agent_lab_measurements.csv  сера/Т95/цетан по ЛИМС и ПАК, с меткой,
                                       когда значение стало известно (available_at)
  reliability_agent_telemetry.csv     длинный горизонт (недели/месяцы) — износ

Дальше эти файлы читает код агентов качества/надёжности — не через этот
скрипт, а напрямую (pandas.read_csv), как обычные готовые датасеты.
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

    print("Готово. Файлы для агентов качества и надёжности:\n")
    for name, path in paths.items():
        size_kb = path.stat().st_size / 1024
        print(f"  {name:26} {path}  ({size_kb:,.0f} КБ)")

    print(
        "\nКороткий горизонт (0-3ч) — quality_agent_telemetry.csv:\n"
        f"  теги: {', '.join(config.tag_key(t, u) for t, u, _ in config.QUALITY_AGENT_TAGS)}\n"
        "\nДлинный горизонт (недели/месяцы) — reliability_agent_telemetry.csv:\n"
        f"  теги: {', '.join(config.tag_key(t, u) for t, u, _ in config.RELIABILITY_AGENT_TAGS)}\n"
        "\nВ обоих файлах есть is_running_AVT и is_running_24-2000 — "
        "не использовать строки, где установка стоит.\n"
        "\nВ quality_agent_lab_measurements.csv использовать available_at, "
        "а не timestamp — иначе утечка из будущего (ЛИМС публикуется с "
        "задержкой 4 часа)."
    )


if __name__ == "__main__":
    main()
