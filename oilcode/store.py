"""Хранилище состояния между тиками — калибровка агентов и лог решений.

ЗАЧЕМ ЭТО ВООБЩЕ НУЖНО

Раньше `Orchestrator.fit()` калибровал агентов один раз при запуске процесса
и больше никогда. Это ломается ровно в тот момент, когда систему запускают
не одноразовым скриптом, а как что-то, что тикает: приходят новые пробы
ЛИМС, стареет катализатор — а откалиброванные пороги застыли на моменте
запуска и никогда не обновятся, пока процесс не перезапустят руками.

Здесь: калибровка каждого агента лежит в SQLite с отметкой времени.
На каждый тик оркестратор спрашивает "не протухла ли калибровка" (см.
config.REFIT_INTERVAL_H) — если нет, использует сохранённую, не считая
заново. Если протухла — пересчитывает и сохраняет новую. Обучение — редко
и по расписанию, а не на каждый тик (это и есть требование ТЗ, что
воспроизводимость должна давать одинаковый результат при одинаковом
состоянии системы: калибровка — часть состояния, не побочный эффект тика).

Заодно здесь лог решений — прямое требование ТЗ: "система должна
сохранять/показывать входные данные, оценки агентов и итоговую
рекомендацию так, чтобы логику решения можно было проверить".

SQLite, а не Postgres, — чтобы для хакатона не нужен был поднятый сервер
БД. Схема ниже без изменений переносится в Postgres (те же типы, тот же
JSON в текстовом поле), когда до этого дойдёт.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

from oilcode import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_state (
    agent       TEXT PRIMARY KEY,
    fitted_at   TEXT NOT NULL,   -- ISO-время калибровки
    params      TEXT NOT NULL    -- JSON
);

CREATE TABLE IF NOT EXISTS decisions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    tick_at             TEXT NOT NULL,   -- момент, на который принято решение
    decided_at          TEXT NOT NULL,   -- момент, когда реально посчитано (для аудита реалтайма)
    status              TEXT NOT NULL,
    problem             TEXT,
    confidence          TEXT,
    explanation         TEXT,
    state_summary       TEXT,   -- JSON
    action              TEXT,   -- JSON
    expected_effect     TEXT,   -- JSON
    constraints_checked TEXT,   -- JSON
    agent_verdicts      TEXT    -- JSON: сырые оценки всех агентов на этот тик
);

CREATE INDEX IF NOT EXISTS idx_decisions_tick_at ON decisions(tick_at);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(obj):
    """dataclass -> dict рекурсивно, остальное — как есть. Для json.dumps."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


class Store:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or config.CACHE / "state.db")
        self.path.parent.mkdir(exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.path)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # ---------------------------------------------------------------
    # Калибровка агентов
    # ---------------------------------------------------------------

    def save_model_state(self, agent: str, params: dict) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO model_state (agent, fitted_at, params) VALUES (?, ?, ?) "
                "ON CONFLICT(agent) DO UPDATE SET fitted_at=excluded.fitted_at, "
                "params=excluded.params",
                (agent, _now_iso(), json.dumps(_to_jsonable(params), ensure_ascii=False)),
            )

    def load_model_state(self, agent: str, max_age_h: float) -> dict | None:
        """Вернуть параметры, если калибровка не старше max_age_h. Иначе None —

        значит, пора переобучать.
        """
        with self._connect() as con:
            row = con.execute(
                "SELECT fitted_at, params FROM model_state WHERE agent = ?", (agent,)
            ).fetchone()
        if row is None:
            return None
        fitted_at = datetime.fromisoformat(row[0])
        age_h = (datetime.now(timezone.utc) - fitted_at).total_seconds() / 3600
        if age_h > max_age_h:
            return None
        return json.loads(row[1])

    # ---------------------------------------------------------------
    # Лог решений
    # ---------------------------------------------------------------

    def save_decision(self, tick_at: datetime, recommendation, agent_verdicts: dict) -> int:
        rec = _to_jsonable(recommendation)
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO decisions "
                "(tick_at, decided_at, status, problem, confidence, explanation, "
                " state_summary, action, expected_effect, constraints_checked, agent_verdicts) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    tick_at.isoformat(), _now_iso(), rec["status"], rec.get("problem"),
                    rec.get("confidence"), rec.get("explanation"),
                    json.dumps(rec.get("state_summary"), ensure_ascii=False),
                    json.dumps(rec.get("action"), ensure_ascii=False),
                    json.dumps(rec.get("expected_effect"), ensure_ascii=False),
                    json.dumps(rec.get("constraints_checked"), ensure_ascii=False),
                    json.dumps(_to_jsonable(agent_verdicts), ensure_ascii=False),
                ),
            )
            return cur.lastrowid

    def last_decision(self) -> dict | None:
        """Последнее по времени тика решение — нужно оркестратору, чтобы

        не дёргаться (см. config.MIN_RECOMMENDATION_REPEAT_H).
        """
        with self._connect() as con:
            row = con.execute(
                "SELECT tick_at, status, action FROM decisions "
                "ORDER BY tick_at DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return {"tick_at": row[0], "status": row[1], "action": json.loads(row[2] or "null")}

    def decisions_since(self, since: datetime):
        with self._connect() as con:
            return con.execute(
                "SELECT tick_at, status, problem, explanation FROM decisions "
                "WHERE tick_at >= ? ORDER BY tick_at", (since.isoformat(),)
            ).fetchall()
