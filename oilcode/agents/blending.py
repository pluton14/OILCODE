"""Агент блендинга.

ЗОНА ОТВЕТСТВЕННОСТИ: <впиши имя>

Подбирает рецептуру товарного ДТ: в каких долях смешать потоки разного
качества и сколько добавить цетаноповышающей присадки, чтобы попасть в
спецификацию КАК МОЖНО ДЕШЕВЛЕ.

ПОЧЕМУ ЭТО ОТДЕЛЬНЫЙ АГЕНТ, А НЕ ЧАСТЬ ОПТИМИЗАТОРА
Норма "сера <= 10 мг/кг" относится к товарному продукту ПОСЛЕ смешения, а не
к выходу гидроочистки (уточнено у организаторов). Поэтому качество на выходе
ГО — это не ограничение, а решение: гидроочистка может выпускать продукт с
серой выше 10, если блендинг сведёт смесь в норму. Отсюда вся экономика.

ГЛАВНЫЙ ЭКОНОМИЧЕСКИЙ РАЗВОРОТ
Чем глубже обессеривание, тем дороже производство. Значит, "выжать серу в
ноль" — это не победа, а убыток. Правильная цель: попасть в спецификацию с
минимальным запасом, который мы считаем безопасным.

ДОПУЩЕНИЕ: данных и схем по блендингу не выдано и не будет. Резервуары в
config.BLENDING — модельные. Организаторы такой сценарий разрешили явно,
при условии что допущения описаны.

ЧТО ДЕЛАТЬ ДАЛЬШЕ:
  v0  вместо перебора с шагом 0.1 — линейное программирование (scipy.linprog),
      задача ровно этой формы
  v1  связать качество резервуаров с реальной историей ГО: брать квантили
      качества за разные периоды вместо выдуманных чисел
  v2  учесть нелинейность смешения для Т95 (по объёму/индексам смешения)
"""

from __future__ import annotations

from itertools import product

from oilcode import config
from oilcode.contracts import BlendingResult, BlendRecipe


class BlendingAgent:
    def __init__(self):
        self.spec = config.PRODUCT_SPEC
        self.econ = config.ECONOMICS
        self.cfg = config.BLENDING

    # ----------------------------------------------------------------------

    def _tank_cost(self, sulfur: float) -> float:
        """Стоимость тонны компонента.

        Чем ниже сера, тем дороже: обессеривание стоит энергии.
        Знак и смысл — со слов организаторов, величина — допущение.
        """
        base = self.econ["base_cost_per_t"]
        removed = self.spec["sulfur_mg_kg_max"] - sulfur
        return base * (1 + self.econ["cost_per_mg_kg_sulfur_removed"] * removed)

    def _evaluate(self, tanks: dict[str, dict], fractions: dict[str, float],
                  improver_pct: float) -> BlendRecipe:
        """Свойства смеси при заданной рецептуре.

        Смешение считаем линейным по массовым долям — стандартное упрощение,
        организаторы разрешили простую модель.
        """
        sulfur = sum(fractions[t] * tanks[t]["sulfur_mg_kg"] for t in fractions)
        t95 = sum(fractions[t] * tanks[t]["t95_c"] for t in fractions)
        cetane = sum(fractions[t] * tanks[t]["cetane"] for t in fractions)
        cetane += improver_pct * self.cfg["cetane_gain_per_pct"]

        fuel_cost = sum(fractions[t] * self._tank_cost(tanks[t]["sulfur_mg_kg"])
                        for t in fractions)
        improver_cost = (improver_pct / 100) * (
            self.econ["cetane_improver_cost_ratio"] * self.econ["base_cost_per_t"]
        )

        recipe = BlendRecipe(
            fractions={k: round(v, 3) for k, v in fractions.items() if v > 0},
            improver_pct=improver_pct,
            blended={"sulfur_mg_kg": round(sulfur, 2),
                     "t95_c": round(t95, 1),
                     "cetane": round(cetane, 2)},
            cost=round(fuel_cost + improver_cost, 4),
        )

        # Жёсткие проверки. Нарушил — выбывает, компенсировать нечем.
        if sulfur > self.spec["sulfur_mg_kg_max"]:
            recipe.violations.append(
                f"сера {sulfur:.2f} > {self.spec['sulfur_mg_kg_max']} мг/кг")
        if t95 > self.spec["t95_max_c"]:
            recipe.violations.append(
                f"Т95 {t95:.1f} > {self.spec['t95_max_c']} C")
        if cetane < self.spec["cetane_min"]:
            recipe.violations.append(
                f"цетановое число {cetane:.2f} < {self.spec['cetane_min']}")
        if improver_pct > config.HARD_LIMITS["cetane_improver_max_pct"]:
            recipe.violations.append(f"присадки {improver_pct}% > 3%")

        recipe.meets_spec = not recipe.violations
        return recipe

    # ----------------------------------------------------------------------

    def solve(self, product_quality: dict[str, float] | None = None) -> BlendingResult:
        """Найти самую дешёвую рецептуру, проходящую спецификацию.

        product_quality — качество текущего потока с гидроочистки. Если
        передано, он участвует в смешении как ещё один компонент: именно
        через него решение по режиму реактора влияет на блендинг.
        """
        tanks = {k: dict(v) for k, v in self.cfg["tanks"].items()}
        if product_quality:
            tanks["current"] = product_quality

        result = BlendingResult()
        names = sorted(tanks)
        step = self.cfg["fraction_step"]
        n_steps = int(round(1 / step))

        improver_options = [
            round(i * self.cfg["improver_step"], 2)
            for i in range(int(config.HARD_LIMITS["cetane_improver_max_pct"]
                               / self.cfg["improver_step"]) + 1)
        ]

        best: BlendRecipe | None = None

        # Перебор целочисленных разбиений единицы на доли резервуаров.
        min_current = self.cfg["min_current_fraction"] if "current" in tanks else 0.0

        for combo in product(range(n_steps + 1), repeat=len(names)):
            if sum(combo) != n_steps:
                continue
            fractions = {n: c / n_steps for n, c in zip(names, combo)}

            # Поток с гидроочистки производится непрерывно — его нельзя
            # просто не использовать, иначе задача вырождается в разбавление.
            if fractions.get("current", 0.0) + 1e-9 < min_current:
                continue

            for improver in improver_options:
                recipe = self._evaluate(tanks, fractions, improver)
                result.considered += 1
                if not recipe.meets_spec:
                    continue
                if best is None or recipe.cost < best.cost:
                    best = recipe

        if best is None:
            result.rejection_reason = (
                "Ни одна рецептура не проходит спецификацию: доступные потоки "
                "слишком плохие по качеству"
            )
            return result

        result.feasible = True
        result.best = best
        margin = self.spec["sulfur_mg_kg_max"] - best.blended["sulfur_mg_kg"]
        result.notes.append(
            f"Запас по сере {margin:.2f} мг/кг. Больший запас означал бы "
            f"переочистку и лишние затраты."
        )
        return result
