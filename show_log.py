"""Смотрелка журнала работы системы — куда глядеть, пока она тикает.

Система пишет каждое решение в cache/state.db (см. oilcode/store.py), но
SQLite руками читать неудобно. Здесь три режима, по нарастанию подробности:

    python show_log.py                 что происходило: строка на тик
    python show_log.py --last          разбор последнего решения целиком
    python show_log.py --tick "2025-06-10 08:00"    разбор конкретного тика
    python show_log.py --models        состояние обученных моделей агентов

Журнал — это то, что ТЗ требует показывать: входные данные, оценки агентов
и итоговую рекомендацию так, чтобы логику решения можно было проверить.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone

from oilcode import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DB = config.CACHE / "state.db"

STATUS_RU = {
    "recommendation": "рекомендация",
    "no_action": "не вмешиваться",
    "refusal": "отказ",
}


def _connect() -> sqlite3.Connection:
    if not DB.exists():
        sys.exit(f"Журнала ещё нет: {DB}\n"
                 f"Сначала прогоните систему: python run_realtime_sim.py")
    return sqlite3.connect(DB)


def show_table(con: sqlite3.Connection, limit: int) -> None:
    rows = con.execute(
        "SELECT tick_at, status, confidence, problem FROM decisions "
        "ORDER BY tick_at DESC LIMIT ?", (limit,)).fetchall()
    if not rows:
        print("Журнал пуст.")
        return
    print(f"{'момент':17} {'решение':15} {'доверие':22} проблема")
    print("-" * 100)
    for tick_at, status, conf, problem in reversed(rows):
        moment = tick_at.replace("T", " ")[:16]
        print(f"{moment:17} {STATUS_RU.get(status, status):15} "
              f"{(conf or '')[:20]:22} {(problem or '')[:45]}")

    total = con.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    print("-" * 100)
    print(f"показано {len(rows)} из {total} решений в журнале")
    by_status = con.execute(
        "SELECT status, COUNT(*) FROM decisions GROUP BY status").fetchall()
    print("итого: " + ", ".join(
        f"{STATUS_RU.get(s, s)} {n}" for s, n in by_status))


def show_one(con: sqlite3.Connection, tick: str | None) -> None:
    if tick:
        row = con.execute(
            "SELECT * FROM decisions WHERE tick_at LIKE ? ORDER BY id DESC LIMIT 1",
            (f"{tick.replace(' ', 'T')}%",)).fetchone()
    else:
        row = con.execute(
            "SELECT * FROM decisions ORDER BY tick_at DESC LIMIT 1").fetchone()
    if row is None:
        sys.exit("Такого тика в журнале нет.")

    cols = [d[0] for d in con.execute("SELECT * FROM decisions LIMIT 1").description]
    rec = dict(zip(cols, row))

    print("=" * 78)
    print(f"ТИК {rec['tick_at'].replace('T', ' ')[:16]}   "
          f"решение: {STATUS_RU.get(rec['status'], rec['status']).upper()}")
    print(f"посчитано системой в {rec['decided_at'][:19]} (реальное время расчёта)")
    print("=" * 78)

    print(f"\nПРОБЛЕМА\n  {rec['problem']}")
    print(f"\nДОВЕРИЕ К ДАННЫМ\n  {rec['confidence']}")

    action = json.loads(rec["action"] or "null")
    if action:
        print("\nЧТО ПРЕДЛАГАЕТСЯ")
        for ch in action:
            print(f"  {ch['tag']}: {ch['current']:g} -> {ch['proposed']:g}")

    effect = json.loads(rec["expected_effect"] or "null")
    if effect:
        print("\nОЖИДАЕМЫЙ ЭФФЕКТ")
        for k, v in effect.items():
            print(f"  {k}: {v}")

    verdicts = json.loads(rec["agent_verdicts"] or "{}")
    _show_verdicts(verdicts)

    print(f"\nОБЪЯСНЕНИЕ\n  {rec['explanation']}")


def _show_verdicts(v: dict) -> None:
    """Оценки агентов — то, из чего сложилось решение."""
    dq = v.get("data_quality")
    if dq:
        print(f"\nАГЕНТ ДАННЫХ: {dq['overall']}")
        running = ", ".join(f"{u}={'работает' if r else 'СТОИТ'}"
                            for u, r in (dq.get("unit_running") or {}).items())
        if running:
            print(f"  установки: {running}")
        for note in (dq.get("notes") or [])[:4]:
            print(f"  - {note}")
        if dq.get("stale_measurements"):
            print(f"  устарело: {', '.join(dq['stale_measurements'][:4])}")

    q = v.get("quality")
    if q:
        print("\nАГЕНТ КАЧЕСТВА")
        for metric, preds in (q.get("forecast") or {}).items():
            print(f"  прогноз {metric}:")
            for p in preds:
                iv = (f"[{p['lo']}; {p['hi']}]" if p.get("lo") is not None
                      else "интервала нет")
                print(f"    {p['horizon_min'] // 60}ч: {p['value']:>8.3f}  "
                      f"{iv:20} увер.{p['confidence']}  {p['basis']}")
        for metric, r in (q.get("spec_risk") or {}).items():
            print(f"  риск {metric}: {r['level']}, запас {r['margin']} мг/кг"
                  f"{', ПРЕВЫШЕНИЕ' if r['exceeds_limit'] else ''}")
        for note in (q.get("notes") or [])[:3]:
            print(f"  - {note}")

    r = v.get("reliability")
    if r:
        print(f"\nАГЕНТ НАДЁЖНОСТИ: {r['severity_class']} "
              f"(индекс {r['severity_index']})")
        print(f"  режим разрешён: {'да' if r['regime_allowed'] else 'НЕТ (вето)'}")
        for f in (r.get("risk_factors") or [])[:5]:
            print(f"  - {f['tag']}: {f['issue']}")

    o = v.get("optimization")
    if o:
        print(f"\nАГЕНТ ОПТИМИЗАЦИИ: "
              f"{'есть допустимые варианты' if o['feasible'] else 'вариантов нет'}")
        print(f"  рассмотрено сценариев: {len(o.get('scenarios') or [])}")
        if o.get("rejection_reason"):
            print(f"  причина отказа: {o['rejection_reason']}")


def show_models(con: sqlite3.Connection) -> None:
    rows = con.execute("SELECT agent, fitted_at, params FROM model_state").fetchall()
    if not rows:
        print("Модели ещё не обучались.")
        return
    now = datetime.now(timezone.utc)
    for agent, fitted_at, params_json in rows:
        age_h = (now - datetime.fromisoformat(fitted_at)).total_seconds() / 3600
        limit = config.REFIT_INTERVAL_H.get(agent)
        stale = f"переобучится при следующем тике (порог {limit}ч)" \
            if limit and age_h > limit else "свежая"
        print(f"\n=== {agent} ===")
        print(f"откалиброван {fitted_at[:19]} ({age_h:.1f}ч назад) — {stale}")

        params = json.loads(params_json)
        if agent == "quality":
            print("\n  ПРОГНОЗ (что будет само собой):")
            print(f"    {'гориз':7}{'инерция':>10}{'модель':>10}{'выигрыш':>10}  применяем")
            for h, m in sorted(params.get("models", {}).items(), key=lambda kv: int(kv[0])):
                print(f"    {h}ч{'':5}{m['mae_persistence']:>10.4f}"
                      f"{m['mae_model']:>10.4f}{m['gain_pct']:>9.2f}%  "
                      f"{'да' if m['use_model'] else 'НЕТ, откат на инерцию'}")
            sens = params.get("sensitivity") or {}
            if sens:
                print("\n  ЧУВСТВИТЕЛЬНОСТЬ (что если вмешаться):")
                print(f"    {'рычаг':10}{'dS/dX':>12}  знак сошёлся  применяем")
                for tag, s in sorted(sens.items()):
                    print(f"    {tag:10}{s['d_sulfur_per_unit']:>12.4f}"
                          f"{'  да' if s['sign_matches_process'] else '  НЕТ':>14}  "
                          f"{'да' if s['use'] else 'НЕТ'}")
        elif agent == "reliability":
            print(f"    {'сигнал':10}{'p50':>10}{'p95':>10}")
            for tag, q in sorted(params.items()):
                if isinstance(q, dict):
                    print(f"    {tag:10}{q.get('p50', 0):>10.3f}{q.get('p95', 0):>10.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Журнал работы системы")
    p.add_argument("--last", action="store_true", help="разбор последнего решения")
    p.add_argument("--tick", help="разбор конкретного тика, напр. \"2025-06-10 08:00\"")
    p.add_argument("--models", action="store_true", help="состояние обученных моделей")
    p.add_argument("-n", type=int, default=30, help="сколько строк показать")
    args = p.parse_args()

    con = _connect()
    try:
        if args.models:
            show_models(con)
        elif args.last or args.tick:
            show_one(con, args.tick)
        else:
            show_table(con, args.n)
    finally:
        con.close()


if __name__ == "__main__":
    main()
