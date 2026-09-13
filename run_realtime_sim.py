"""Симуляция реальной работы системы — тикает по историческим меткам времени.

Это не run_demo.py (один момент, для разового показа), а прогон, который
имитирует то, как система будет вести себя в реальной эксплуатации: идёт по
времени с шагом, на каждом тике опрашивает агентов и пишет решение в
oilcode/store.py (SQLite). Калибровка агентов не пересчитывается на каждом
тике — только когда протухла (см. config.REFIT_INTERVAL_H), это и есть
проверка того, что разделение "обучение — оффлайн, тик — онлайн" реально
работает, а не просто написано в комментарии.

    python run_realtime_sim.py --start "2024-06-01" --end "2024-06-05" --step-min 60

Без --end показывает --ticks тиков от --start.
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from oilcode import config
from oilcode.orchestrator import Orchestrator

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

config.require_data_files()


def main() -> None:
    p = argparse.ArgumentParser(description="Симуляция тикающей работы системы")
    p.add_argument("--start", default="2024-06-01 00:00")
    p.add_argument("--end", default=None)
    p.add_argument("--ticks", type=int, default=20, help="сколько тиков, если нет --end")
    p.add_argument("--step-min", type=int, default=60,
                   help="шаг между тиками, минут (по умолчанию верх диапазона "
                        "config.RECOMMENDATION_STEP_MIN)")
    args = p.parse_args()

    start = pd.Timestamp(args.start)
    step = pd.Timedelta(minutes=args.step_min)
    if args.end:
        ticks = pd.date_range(start, pd.Timestamp(args.end), freq=step)
    else:
        ticks = pd.date_range(start, periods=args.ticks, freq=step)

    orchestrator = Orchestrator()
    # Специально НЕ форсируем fit() здесь: на пустой базе decide() сам
    # обучится на первом тике (load_model_state вернёт None), а если база
    # уже есть с прошлого запуска — использует её. Форсировать fit() тут
    # означало бы каждый раз затирать честную проверку "протухла ли
    # калибровка", а не пропускать вызов.

    print(f"Тиков: {len(ticks)}, шаг {args.step_min} мин, "
         f"с {ticks[0]} по {ticks[-1]}\n")
    print(f"{'момент':17} {'статус':15} {'калибровка':11} {'проблема'}")

    refit_count = 0
    t0 = time.time()
    for ts in ticks:
        was_cached = orchestrator.store.load_model_state(
            "reliability", config.REFIT_INTERVAL_H["reliability"]) is not None

        rec = orchestrator.decide(ts)

        refit_mark = "кэш" if was_cached else "ПЕРЕСЧЁТ"
        if not was_cached:
            refit_count += 1
        problem = (rec.problem or "")[:60]
        print(f"{ts:%Y-%m-%d %H:%M} {rec.status:15} {refit_mark:11} {problem}")

    dt = time.time() - t0
    print(f"\nГотово за {dt:.1f} c. Переобучений: {refit_count} из {len(ticks)} тиков.")
    print(f"Лог решений: {orchestrator.store.path}")


if __name__ == "__main__":
    main()
