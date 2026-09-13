"""Агент качества.

ЗОНА ОТВЕТСТВЕННОСТИ: <впиши имя>

РОЛЬ ИЗ ТЗ: оценивает текущее и прогнозное качество, выявляет риск выхода за
спецификацию, оценивает влияние изменения режима.

===========================================================================
ГЛАВНОЕ, ЧТО НАДО ПОНЯТЬ ПРО ЭТОГО АГЕНТА: ВНУТРИ ДВЕ РАЗНЫЕ МОДЕЛИ
===========================================================================

Это не усложнение ради красоты. "Что будет через час" и "что будет, если
поднять температуру" — на этом процессе РАЗНЫЕ вопросы, и отвечать на них
одной моделью нельзя. Мы это проверили на данных, вот результат.

1. ПРОГНОЗ — "что будет, если ничего не трогать".
   Побеждает возврат к среднему: сера ходит вокруг уставки, и если она
   сейчас отклонилась, то вернётся. Режимные теги в прогноз НЕ помогают —
   проверено, с ними становится хуже (+11.8% против +17.6% на 3ч).
   Сравнение с инерцией на валидации (MAE, %, больше — лучше):
        1ч   -23%   инерция побеждает, модель не применяется
        2ч    +3%
        3ч   +18%
   На часе ничто не бьёт инерцию — так и должно быть: процесс под
   замкнутым управлением, за час он никуда не уходит. Мы это не прячем,
   а прямо возвращаем инерцию (см. use_model).

2. ЧУВСТВИТЕЛЬНОСТЬ — "что будет, если поднять T5 на 2 градуса".
   Здесь прогнозная модель бесполезна в принципе: в истории температуру
   поднимают ПОТОМУ ЧТО сера выросла, поэтому мгновенная связь показывает
   обратный знак от физического. Причинный эффект виден только на медленном
   масштабе — суточные средние внутри одного цикла катализатора:
        dS/dT5  = -0.033 мг/кг на градус  (горячее -> чище)
        dS/dF26 = +0.011 мг/кг на м3/ч    (больше нагрузка -> грязнее)
   Знаки совпали с технологией, и это единственная причина им доверять:
   величины малы и неустойчивы между циклами. Коэффициент, у которого знак
   разошёлся с ожидаемым (config.EXPECTED_SULFUR_SIGN), не применяется —
   это признак, что поймали не причину, а сопутствующий дрейф.

ПОЧЕМУ МОДЕЛИ ПРОСТЫЕ (гребневая регрессия, не бустинг):
  1. веса — это несколько чисел, они кладутся в store и переживают
     перезапуск, а не тянут файл модели на десятки мегабайт;
  2. веса читаемы: видно, какой тег и куда двигает серу, и это сверяется
     с технологией, а не принимается на веру;
  3. при связи режима с серой на уровне 0.2 бустинг переобучится раньше,
     чем принесёт пользу. Усложнять осмысленно после того, как простое
     заработает, а не до.

ЧЕГО АГЕНТ НЕ ДЕЛАЕТ: не решает, менять ли режим. Он говорит, что будет с
качеством и насколько близко к границе. Решение — за оркестратором.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oilcode import config
from oilcode.contracts import (
    Change,
    Prediction,
    ProcessState,
    QualityAssessment,
    SpecRisk,
)

# Окна возврата к среднему, часы. Два масштаба: быстрое отклонение от
# последних часов и медленное от суток. Подобраны сравнением на валидации,
# не угаданы.
REVERSION_WINDOWS_H = (6, 24)

# Сила регуляризации для гребневой регрессии.
RIDGE_ALPHA = 10.0

# Минимум суток для оценки структурного коэффициента чувствительности.
MIN_DAYS_FOR_SENSITIVITY = 60


class QualityAgent:
    """Прогноз качества и риск выхода за спецификацию.

    fit()    — оффлайн, редко: калибровка на истории.
    assess() — онлайн, на каждый тик: подстановка в готовые веса.
    """

    def __init__(self, horizon_min: int = 60):
        self.horizon_min = horizon_min
        # Всё обученное состояние JSON-сериализуемо и переживает перезапуск
        # процесса через oilcode/store.py.
        self.models: dict[int, dict] = {}      # прогноз, по горизонтам
        self.sensitivity: dict[str, dict] = {}  # чувствительность к рычагам
        self.feature_names: list[str] = []

    # ==================================================================
    # ОФФЛАЙН: калибровка
    # ==================================================================

    def fit(self, frame: pd.DataFrame | None = None, telemetry=None,
            lims=None) -> "QualityAgent":
        """Калибровка на истории: прогнозные модели + чувствительность.

        frame — подготовленная таблица от Агента данных. Если не передана,
        запрашивается сама: агент качества не читает сырые файлы, он
        спрашивает Агента данных.
        """
        if frame is None:
            from oilcode.agents.data import DataAgent
            frame = DataAgent.training_frame()

        frame = frame.sort_index()
        steps_per_h = self._steps_per_hour(frame.index)
        target = pd.to_numeric(frame["sulfur_mg_kg"], errors="coerce")

        self._fit_forecast(target, steps_per_h)
        self._fit_sensitivity(frame, target)
        return self

    def _fit_forecast(self, target: pd.Series, steps_per_h: float) -> None:
        """Прогноз "если ничего не трогать" — возврат к среднему."""
        X, self.feature_names = self._features(target, steps_per_h)
        train_end = pd.Timestamp(config.TIME_SPLIT["train_end"])
        valid_end = pd.Timestamp(config.TIME_SPLIT["valid_end"])

        self.models = {}
        for h in config.QUALITY_HORIZONS_H:
            # Что реально произошло через h часов, минус то, что предсказала
            # бы инерция. Учим модель именно ошибке инерции.
            y = target.shift(-int(round(h * steps_per_h))) - target

            ok = X.notna().all(axis=1) & y.notna()
            idx = X.index[ok]
            tr = idx[idx <= train_end]
            va = idx[(idx > train_end) & (idx <= valid_end)]
            if len(tr) < 500 or len(va) < 100:
                continue

            w, mu, sd = self._fit_ridge(X.loc[tr].to_numpy(float),
                                        y.loc[tr].to_numpy(float))
            pred = self._apply(X.loc[va].to_numpy(float), w, mu, sd)
            true = y.loc[va].to_numpy(float)
            mae_model = float(np.mean(np.abs(true - pred)))
            # Инерция предсказывает "изменения не будет", то есть ноль.
            mae_persist = float(np.mean(np.abs(true)))
            gain = (mae_persist - mae_model) / mae_persist * 100 if mae_persist else 0.0

            self.models[h] = {
                "weights": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
                "gain_pct": round(gain, 2),
                "mae_model": round(mae_model, 4),
                "mae_persistence": round(mae_persist, 4),
                "resid_std": float(np.std(true - pred)),
                "persist_std": float(np.std(true)),
                # Модель применяется ТОЛЬКО если обыграла инерцию на
                # валидации. Решает проверка, а не наше желание.
                "use_model": bool(gain > 0),
                "n_train": int(len(tr)), "n_valid": int(len(va)),
            }

    def _fit_sensitivity(self, frame: pd.DataFrame, target: pd.Series) -> None:
        """Чувствительность серы к рычагам — структурная, на суточных средних.

        ПОЧЕМУ НЕ НА 10-МИНУТНЫХ: на быстром масштабе связь перевёрнута
        обратной причинностью — оператор поднимает температуру в ответ на
        рост серы. Мгновенная корреляция T5 с серой -0.018, то есть шум.
        Причинный эффект проступает только на суточных средних внутри
        одного цикла катализатора (замена катализатора сбрасывает уровень
        и смешала бы два разных состояния в одно облако точек).
        """
        cycle = self._longest_cycle(frame.index)
        if cycle is None:
            return
        start, end = cycle

        levers = [t for t in config.EXPECTED_SULFUR_SIGN if t in frame.columns]
        if not levers:
            return

        daily = frame.loc[start:end, levers].apply(
            pd.to_numeric, errors="coerce").resample("D").mean()
        daily["_target"] = target.loc[start:end].resample("D").mean()
        daily = daily.dropna()
        if len(daily) < MIN_DAYS_FOR_SENSITIVITY:
            return

        # Все рычаги в одной регрессии: иначе эффект нагрузки припишется
        # температуре, потому что они меняются вместе.
        X = daily[levers].to_numpy(float)
        w, mu, sd = self._fit_ridge(X, daily["_target"].to_numpy(float))
        # Веса стандартизованы — возвращаем в исходные единицы.
        raw = w[:-1] / sd

        self.sensitivity = {}
        for tag, coef in zip(levers, raw):
            expected = config.EXPECTED_SULFUR_SIGN[tag]
            sign_ok = bool(np.sign(coef) == np.sign(expected)) if coef else False
            self.sensitivity[tag] = {
                "d_sulfur_per_unit": float(coef),
                "expected_sign": int(expected),
                "sign_matches_process": sign_ok,
                # Применяем только то, что согласуется с технологией.
                # Обратный знак — признак, что поймали реакцию оператора,
                # а не физику процесса.
                "use": sign_ok,
                "n_days": int(len(daily)),
                "cycle": [str(start.date()), str(end.date())],
            }

    @staticmethod
    def _longest_cycle(index: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Самый длинный промежуток между заменами катализатора.

        Границы — config.KNOWN_CATALYST_RESETS плюс концы ряда.
        """
        if len(index) == 0:
            return None
        bounds = [index.min()] + [pd.Timestamp(d) for d in
                                  config.KNOWN_CATALYST_RESETS] + [index.max()]
        bounds = sorted(b for b in bounds if index.min() <= b <= index.max())
        best, best_len = None, pd.Timedelta(0)
        for a, b in zip(bounds, bounds[1:]):
            if b - a > best_len:
                best, best_len = (a, b), b - a
        return best

    # --- состояние для store ------------------------------------------

    def state_dict(self) -> dict:
        return {"models": {str(k): v for k, v in self.models.items()},
                "sensitivity": self.sensitivity,
                "feature_names": self.feature_names}

    def load_state(self, state: dict) -> "QualityAgent":
        self.models = {int(k): v for k, v in state.get("models", {}).items()}
        self.sensitivity = state.get("sensitivity", {})
        self.feature_names = state.get("feature_names", [])
        return self

    # ==================================================================
    # ОНЛАЙН: оценка на тике
    # ==================================================================

    def assess(self, state: ProcessState) -> QualityAssessment:
        out = QualityAssessment(data_confidence=state.data_quality.overall)

        if state.data_quality.overall == "insufficient":
            out.notes.append("Данных недостаточно для оценки качества")
            return out

        # Текущие значения показателей — как есть, с доверием по источнику.
        for metric, meas in state.quality.items():
            base = {"LIMS": 0.95, "PAK": 0.85, "VAK": 0.6}[meas.source]
            age_penalty = min((meas.age_hours or 0) / 48.0, 1.0) * 0.4
            out.predictions[metric] = Prediction(
                value=meas.value,
                confidence=round(max(base - age_penalty, 0.1), 2),
                horizon_min=0, basis=f"текущее значение, {meas.source}",
            )
            if meas.age_hours and meas.age_hours > 24:
                out.notes.append(
                    f"{metric}: последнее значение получено {meas.age_hours:.0f}ч "
                    f"назад ({meas.source}) — прогноз ненадёжен"
                )

        sulfur = state.quality.get(config.TARGET_METRIC)
        if sulfur is None:
            out.notes.append("Нет серы — прогнозировать нечего")
            return out

        forecasts = self._forecast(state, sulfur.value)
        out.forecast[config.TARGET_METRIC] = forecasts
        for h, m in self.models.items():
            out.model_gain_pct[h] = m["gain_pct"]

        if forecasts:
            main_h = self.horizon_min / 60
            out.predictions[config.TARGET_METRIC] = min(
                forecasts, key=lambda p: abs(p.horizon_min / 60 - main_h))

        # ВНИМАНИЕ: сера на выходе гидроочистки выше 10 мг/кг — это НЕ авария.
        # Норма 10 мг/кг относится к товарному продукту ПОСЛЕ блендинга
        # (уточнено у организаторов). На точке "Гидроочистка 2" превышения
        # занимают 14.7% замеров, и это штатная картина.
        # Поэтому риск считается относительно порога, выше которого блендинг
        # уже не сможет вытянуть смесь в спецификацию.
        out.spec_risk[config.TARGET_METRIC] = self._risk(
            config.TARGET_METRIC, sulfur.value, forecasts,
            config.BLENDING["max_rescuable_sulfur_mg_kg"])
        return out

    # ==================================================================
    # ВЛИЯНИЕ ИЗМЕНЕНИЯ РЕЖИМА — то, что спрашивает агент оптимизации
    # ==================================================================

    def predict_for_changes(self, state: ProcessState,
                            changes: list[Change]) -> dict:
        """Как изменится сера, если применить changes.

        Считается по СТРУКТУРНЫМ коэффициентам (медленная связь), а не по
        прогнозной модели: прогнозная обучена на "что будет само собой" и
        для вопроса "что будет, если вмешаться" даёт неверный знак.

        ЧЕСТНЫЕ ОГОВОРКИ, которые обязан учитывать агент оптимизации:
          * эффект структурный, то есть проявляется за часы-сутки, а не
            мгновенно. Для решения "прямо сейчас" это верхняя оценка;
          * коэффициенты оценены на одном цикле катализатора и малы —
            экстраполировать далеко за исторический диапазон нельзя;
          * рычаги, у которых знак разошёлся с технологией, не применяются
            вовсе и попадают в ignored.
        """
        sulfur = state.quality.get(config.TARGET_METRIC)
        if sulfur is None:
            return {}

        delta = 0.0
        applied, ignored = [], []
        for ch in changes:
            s = self.sensitivity.get(ch.tag)
            if s is None:
                ignored.append(f"{ch.tag}: нет оценки чувствительности")
                continue
            if not s["use"]:
                ignored.append(
                    f"{ch.tag}: знак коэффициента разошёлся с технологией, "
                    "не применяем")
                continue
            contrib = s["d_sulfur_per_unit"] * ch.delta
            delta += contrib
            applied.append(f"{ch.tag} {ch.delta:+g} -> сера {contrib:+.3f}")

        return {
            "sulfur_now": sulfur.value,
            "sulfur_after": round(sulfur.value + delta, 3),
            "delta": round(delta, 4),
            "applied": applied,
            "ignored": ignored,
            "basis": "структурные коэффициенты на суточных средних",
        }

    # ==================================================================
    # Внутреннее
    # ==================================================================

    def _forecast(self, state: ProcessState, anchor: float) -> list[Prediction]:
        """Прогноз на каждый горизонт: инерция плюс поправка, если она

        доказала пользу на валидации.
        """
        preds: list[Prediction] = []
        for h in config.QUALITY_HORIZONS_H:
            model = self.models.get(h)
            delta, spread = 0.0, None
            basis = "инерция (модель не обыграла её)"

            if model is None:
                basis = "инерция (модель не откалибрована)"
            elif not model["use_model"]:
                spread = model["persist_std"]
            else:
                x = self._feature_row(state)
                if x is None:
                    basis = "инерция (не хватило истории для признаков)"
                    spread = model["persist_std"]
                else:
                    delta = float(self._apply(
                        x[None, :], np.array(model["weights"]),
                        np.array(model["mu"]), np.array(model["sd"]))[0])
                    basis = (f"возврат к среднему "
                             f"({model['gain_pct']:+.1f}% к MAE против инерции)")
                    spread = model["resid_std"]

            value = anchor + delta
            lo = hi = None
            if spread is not None:
                lo, hi = round(value - spread, 3), round(value + spread, 3)
            preds.append(Prediction(
                value=round(value, 3),
                confidence=round(self._confidence(model, state), 2),
                horizon_min=h * 60, lo=lo, hi=hi, basis=basis,
            ))
        return preds

    @staticmethod
    def _confidence(model: dict | None, state: ProcessState) -> float:
        base = 0.5 if model is None or not model["use_model"] else 0.7
        if state.data_quality.overall == "degraded":
            base -= 0.15
        return max(base, 0.1)

    @staticmethod
    def _risk(metric: str, current: float, forecasts: list[Prediction],
              limit: float) -> SpecRisk:
        """Риск считаем по ХУДШЕМУ из прогнозов и по ВЕРХНЕЙ границе

        интервала, а не по точечному значению. Сказать "запас 0.3 мг/кг",
        умолчав, что разброс модели ±2, — это ложное спокойствие.
        """
        worst = current
        for p in forecasts:
            worst = max(worst, p.hi if p.hi is not None else p.value)
        margin = limit - worst
        return SpecRisk(
            metric=metric, exceeds_limit=worst > limit,
            margin=round(margin, 2),
            level="high" if margin < 1 else "medium" if margin < 3 else "low",
        )

    # --- признаки прогноза ---------------------------------------------

    @staticmethod
    def _steps_per_hour(index: pd.DatetimeIndex) -> float:
        """Шаг сетки выводим из самих данных, а не хардкодим 10 минут:

        если выгрузка однажды придёт с другим шагом, лаги не должны молча
        начать означать другое время.
        """
        step = pd.Series(index).diff().median()
        if pd.isna(step) or step.total_seconds() <= 0:
            return 6.0
        return 3600.0 / step.total_seconds()

    @classmethod
    def _features(cls, target: pd.Series,
                  steps_per_h: float) -> tuple[pd.DataFrame, list[str]]:
        """Отклонение серы от её же скользящего среднего на двух масштабах.

        Режимные теги сюда СОЗНАТЕЛЬНО не входят: проверено на валидации,
        с ними прогноз становится хуже (+11.8% против +17.6% на 3ч). Они
        нужны в другом месте — в оценке чувствительности.
        """
        cols = {}
        for wh in REVERSION_WINDOWS_H:
            n = int(round(wh * steps_per_h))
            cols[f"dev{wh}h"] = target - target.rolling(
                n, min_periods=max(n // 3, 2)).mean()
        X = pd.DataFrame(cols, index=target.index)
        return X, list(X.columns)

    def _feature_row(self, state: ProcessState) -> np.ndarray | None:
        """Тот же вектор признаков, но на одном моменте — из окна истории,

        которое приложил Агент данных.
        """
        hist = state.target_history
        if hist is None or len(hist) == 0 or not self.feature_names:
            return None
        target = pd.to_numeric(pd.Series(hist), errors="coerce")
        X, _ = self._features(target, self._steps_per_hour(target.index))
        X = X.reindex(columns=self.feature_names)
        if X.empty:
            return None
        row = X.iloc[-1]
        if row.isna().any():
            return None
        return row.to_numpy(float)

    # --- гребневая регрессия на numpy ----------------------------------

    @staticmethod
    def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA):
        """Стандартизация + L2. Без sklearn: пятнадцать строк, читаемо,

        и не тянет зависимость ради одной формулы.
        """
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])  # + свободный член
        penalty = alpha * np.eye(Z.shape[1])
        penalty[-1, -1] = 0.0  # свободный член не штрафуем
        w = np.linalg.solve(Z.T @ Z + penalty, Z.T @ y)
        return w, mu, sd

    @staticmethod
    def _apply(X: np.ndarray, w: np.ndarray, mu: np.ndarray,
               sd: np.ndarray) -> np.ndarray:
        Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
        return Z @ w
