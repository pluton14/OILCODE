"""Печать рекомендации оператору — семь блоков из ТЗ.

Оператор не читает JSON. Здесь Recommendation превращается в текст, который
можно показать человеку и защитить перед жюри.
"""

from __future__ import annotations

from oilcode.contracts import Recommendation

WIDTH = 78

STATUS_TITLE = {
    "recommendation": "РЕКОМЕНДАЦИЯ",
    "no_action": "ВМЕШАТЕЛЬСТВО НЕ ТРЕБУЕТСЯ",
    "refusal": "ОТКАЗ ОТ РЕКОМЕНДАЦИИ",
}


def _block(title: str, body: str) -> str:
    return f"\n{title}\n{'-' * WIDTH}\n{body}"


def render(rec: Recommendation) -> str:
    parts = [
        "=" * WIDTH,
        f"  {STATUS_TITLE[rec.status]}",
        "=" * WIDTH,
    ]

    parts.append(_block(
        "1. ВРЕМЯ И СОСТОЯНИЕ",
        "\n".join(f"  {k:<16} {v}" for k, v in rec.state_summary.items()),
    ))

    parts.append(_block("2. ПРОБЛЕМА / РИСК", f"  {rec.problem}"))

    if rec.action:
        body = "\n".join(
            f"  {c.tag}: {c.current:g} -> {c.proposed:g}  ({c.delta:+.3g})"
            for c in rec.action
        )
    else:
        body = "  действий не предлагается"
    parts.append(_block("3. ПРЕДЛАГАЕМОЕ ДЕЙСТВИЕ", body))

    if rec.expected_effect:
        parts.append(_block(
            "4. ОЖИДАЕМЫЙ ЭФФЕКТ",
            "\n".join(f"  {k:<20} {v}" for k, v in rec.expected_effect.items()),
        ))

    if rec.constraints_checked:
        parts.append(_block(
            "5. ПРОВЕРКА ОГРАНИЧЕНИЙ",
            "\n".join(f"  [x] {c}" for c in rec.constraints_checked),
        ))

    parts.append(_block("6. УВЕРЕННОСТЬ", f"  {rec.confidence}"))
    parts.append(_block("7. ОБЪЯСНЕНИЕ", f"  {rec.explanation}"))

    if rec.alternatives:
        rows = []
        for s in rec.alternatives:
            mark = "OK " if s.hard_constraints_passed else "NO "
            reason = "" if s.hard_constraints_passed else "  <- " + "; ".join(s.rejected_by)
            rows.append(f"  {mark}{s.id:<12} score={s.score:<8.3f}{reason}")
        parts.append(_block("   АЛЬТЕРНАТИВЫ", "\n".join(rows)))

    parts.append("=" * WIDTH)
    return "\n".join(parts)
