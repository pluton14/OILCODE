"""Обратная совместимость: build_process_state = DataAgent().build_state.

Сама логика — в oilcode/agents/data.py (класс DataAgent), рядом с остальными
агентами. Этот модуль оставлен, чтобы старый импорт
`from oilcode.data.state import build_process_state` не ломался (используется
в тестах).
"""

from __future__ import annotations

from datetime import datetime

from oilcode.agents.data import DataAgent
from oilcode.contracts import ProcessState

_agent = DataAgent()


def build_process_state(timestamp: str | datetime) -> ProcessState:
    return _agent.build_state(timestamp)
