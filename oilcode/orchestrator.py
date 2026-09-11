"""Оркестратор.

ЗОНА ОТВЕТСТВЕННОСТИ: капитан

Ничего не считает сам. Спрашивает агентов, разрешает конфликт целей и
упаковывает результат в понятный оператору ответ — либо в обоснованный отказ.

Приоритет при конфликте (прямо из ТЗ): качество и жёсткие ограничения выше
экономики. Если агент надёжности против, а агент оптимизации нашёл выгодный
вариант — побеждает агент надёжности.
"""

from __future__ import annotations

from datetime import datetime

from oilcode import config
from oilcode.agents.blending import BlendingAgent
from oilcode.agents.optimization import OptimizationAgent
from oilcode.agents.quality import QualityAgent
from oilcode.agents.reliability import ReliabilityAgent
from oilcode.contracts import ProcessState, Recommendation
from oilcode.data.state import build_process_state
from oilcode.data import loaders


class Orchestrator:
    def __init__(self):
        self.quality_agent = QualityAgent()
        self.reliability_agent = ReliabilityAgent()
        self.blending_agent = BlendingAgent()
        self.optimization_agent = OptimizationAgent(blending=self.blending_agent)
        self._fitted = False

    def fit(self) -> "Orchestrator":
        """Калибровка агентов на истории. Делается один раз при запуске."""
        telemetry = loaders.load_telemetry()
        self.quality_agent.fit(telemetry=telemetry)
        self.reliability_agent.fit(telemetry=telemetry)
        self._fitted = True
        return self

    def decide(self, timestamp: str | datetime) -> Recommendation:
        if not self._fitted:
            self.fit()

        state = build_process_state(timestamp)
        return self.decide_for_state(state)

    def decide_for_state(self, state: ProcessState) -> Recommendation:
        rec = Recommendation(timestamp=state.timestamp, status="refusal")
        rec.state_summary = self._summarize(state)

        # --- Шаг 1: данные вообще пригодны? --------------------------------
        if state.data_quality.overall == "insufficient":
            rec.problem = "Недостаточно данных для принятия решения"
            rec.confidence = "нет доверия"
            rec.explanation = (
                "Надёжной рекомендации нет: " + "; ".join(state.data_quality.notes)
                + ". Рискованное предложение в такой ситуации хуже, чем его отсутствие."
            )
            return rec

        # --- Шаг 2: опрашиваем агентов -------------------------------------
        quality = self.quality_agent.assess(state)
        reliability = self.reliability_agent.assess(state)
        optimization = self.optimization_agent.optimize(state, quality, reliability)

        rec.constraints_checked = [
            f"товарный продукт: сера <= {config.PRODUCT_SPEC['sulfur_mg_kg_max']} мг/кг",
            f"товарный продукт: Т95 <= {config.PRODUCT_SPEC['t95_max_c']} C",
            f"товарный продукт: цетановое число >= {config.PRODUCT_SPEC['cetane_min']}",
            "доли компонентов блендинга в сумме 100%, присадка не более 3%",
            "изменения в пределах модельных ограничений агента надёжности",
        ]
        rec.confidence = self._describe_confidence(state, quality)

        # --- Шаг 3: есть ли проблема вообще? -------------------------------
        # Напоминание: сера на выходе ГО выше 10 мг/кг сама по себе НЕ авария.
        # Норма относится к товарному продукту после блендинга.
        risk = quality.spec_risk.get(config.TARGET_METRIC)
        sulfur_now = state.quality.get(config.TARGET_METRIC)
        problems = []

        if risk and risk.exceeds_limit:
            problems.append(
                f"сера на выходе ГО {sulfur_now.value:.2f} мг/кг — блендинг уже "
                f"не вытянет смесь в спецификацию"
            )
        elif risk and risk.level == "high":
            problems.append(
                f"запас до предела смешиваемости всего {risk.margin:.2f} мг/кг"
            )

        # Переочистка — это тоже проблема, только экономическая: лишняя
        # глубина обессеривания оплачена энергией, а спецификация её не
        # требует. Судим не по произвольному порогу серы, а по деньгам:
        # насколько дешевле стала бы смесь при лучшем режиме.
        saving = 0.0
        if optimization.feasible and optimization.baseline_score is not None:
            best_score = optimization.best.score if optimization.best else 0.0
            saving = best_score - optimization.baseline_score
            if saving > config.ECONOMICS["min_saving_to_act"]:
                problems.append(
                    f"режим дороже необходимого: смягчение сэкономит "
                    f"{saving:.4f} усл.ед./т при сохранении спецификации"
                )

        if reliability.severity_class != "normal":
            problems.append(
                f"тяжёлый режим (индекс {reliability.severity_index}): "
                + "; ".join(f.issue for f in reliability.risk_factors)
            )
        rec.problem = "; ".join(problems) if problems else "отклонений не обнаружено"

        # --- Шаг 4: конфликт целей -----------------------------------------
        if not reliability.regime_allowed:
            rec.status = "refusal"
            rec.explanation = (
                "Агент надёжности запретил изменение режима: оборудование уже "
                "работает за пределами исторически наблюдавшегося диапазона. "
                "Выигрыш по качеству или выпуску не оправдывает риск."
            )
            return rec

        if not optimization.feasible:
            rec.status = "refusal"
            rec.explanation = (
                f"Надёжной рекомендации нет: {optimization.rejection_reason}. "
                "Отказ здесь — корректный ответ, а не сбой системы."
            )
            rec.alternatives = optimization.scenarios[:5]
            return rec

        # --- Шаг 5: нужно ли вообще вмешиваться? ---------------------------
        if not problems:
            rec.status = "no_action"
            rec.explanation = (
                "Процесс в норме, запас по спецификации достаточный, режим не "
                "тяжёлый. Лишнее управляющее воздействие только внесёт возмущение."
            )
            return rec

        # --- Шаг 6: рекомендация -------------------------------------------
        best = optimization.best
        rec.status = "recommendation"
        rec.action = best.changes

        # Показываем обе стадии: что будет на выходе ГО и что получится
        # в товарном продукте после смешения.
        blend = self.blending_agent.solve({
            "sulfur_mg_kg": best.predicted_quality.get(config.TARGET_METRIC),
            "t95_c": best.predicted_quality.get("t95_c", 357.0),
            "cetane": best.predicted_quality.get("cetane", 51.0),
        })
        rec.expected_effect = {
            "сера на выходе ГО": f"{best.predicted_quality.get(config.TARGET_METRIC)} мг/кг",
            "выпуск": f"{best.throughput_delta_pct:+.1f}%",
            "энергия (прокси)": f"{best.energy_proxy_delta:+.1f}%",
            "риск оборудования": f"индекс {best.predicted_severity}",
        }
        if blend.feasible:
            b = blend.best
            recipe = ", ".join(f"{k} {v:.0%}" for k, v in b.fractions.items())
            rec.expected_effect["товарный продукт"] = (
                f"сера {b.blended['sulfur_mg_kg']} мг/кг, "
                f"Т95 {b.blended['t95_c']} C, цетан {b.blended['cetane']}"
            )
            rec.expected_effect["рецептура блендинга"] = recipe
            if b.improver_pct:
                rec.expected_effect["цетаноповышающая присадка"] = f"{b.improver_pct}%"
            rec.expected_effect["стоимость смеси"] = f"{b.cost:.4f} усл.ед./т"
        rejected = [s for s in optimization.scenarios if not s.hard_constraints_passed]
        rec.alternatives = [s for s in optimization.scenarios
                            if s.hard_constraints_passed and s.id != best.id][:3]
        rec.explanation = (
            f"Из {len(optimization.scenarios)} рассмотренных вариантов "
            f"{len(rejected)} отброшено по жёстким ограничениям. "
            f"Выбран «{best.id}»: среди прошедших спецификацию он даёт "
            f"наименьшую суммарную стоимость с учётом платы за риск для "
            f"катализатора. Больший запас по сере здесь не нужен — он означал "
            f"бы переочистку за счёт лишних энергозатрат."
        )
        return rec

    @staticmethod
    def _summarize(state: ProcessState) -> dict[str, str]:
        summary = {"момент": state.timestamp.strftime("%Y-%m-%d %H:%M")}
        for metric, m in sorted(state.quality.items()):
            age = f"{m.age_hours:.1f}ч назад" if m.age_hours is not None else "возраст неизвестен"
            summary[metric] = f"{m.value:g} ({m.source}, {age})"
        return summary

    @staticmethod
    def _describe_confidence(state, quality) -> str:
        level = {"ok": "высокая", "degraded": "средняя", "insufficient": "нет"}[
            state.data_quality.overall
        ]
        if state.data_quality.stale_measurements:
            return f"{level} — устарело: {', '.join(state.data_quality.stale_measurements[:3])}"
        return level
