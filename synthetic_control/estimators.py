import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy.optimize import fmin_slsqp
from functools import partial
from .base import SyntheticControl
from scipy.stats import norm
from scipy.stats import norm, t

class ClassicSyntheticControl(SyntheticControl):
    """
    Классическая реализация метода Synthetic Control.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame с данными
    metric : str
        Название метрики для анализа
    period_index : str
        Название колонки с временными периодами
    shopno : str
        Название колонки с идентификаторами магазинов
    treated : str
        Название колонки, указывающей на обработанные единицы
    after_treatment : str
        Название колонки, указывающей на периоды после вмешательства
    bootstrap_rounds : int, default=100
        Количество раундов бутстрепа для оценки стандартной ошибки
    seed : int, default=42
        Seed для воспроизводимости результатов
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights_ = None
        self.control_units_ = None
        self.X_ = None
        self.y_ = None
        
    def fit(self) -> None:
        """
        Обучение модели Classic Synthetic Control.
        """
        # Подготовка данных для предварительного периода
        df_pre = self.data[~self.data[self.after_treatment]]
        
        # Получение данных для обработанной единицы
        treated_data = df_pre[df_pre[self.treated]]
        if len(treated_data) == 0:
            raise ValueError("Нет данных для обработанной единицы в предварительном периоде")
        
        # Получение данных для контрольных единиц
        control_data = df_pre[~df_pre[self.treated]]
        if len(control_data) == 0:
            raise ValueError("Нет данных для контрольных единиц")
        
        # Преобразование в матричный формат
        df_pre_control = control_data.pivot(
            index=self.period_index,
            columns=self.shopno,
            values=self.metric
        )
        
        self.control_units_ = list(df_pre_control.columns)
        self.X_ = df_pre_control.values
        
        self.y_ = treated_data.groupby(self.period_index)[self.metric].mean().values
        
        # Проверка размерностей
        if self.X_.shape[0] != len(self.y_):
            raise ValueError(
                f"Несоответствие размерностей: X_.shape[0]={self.X_.shape[0]}, "
                f"len(y_)={len(self.y_)}"
            )
        
        # Инициализация весов
        n_features = self.X_.shape[1]
        init_w = np.ones(n_features) / n_features
        
        # Ограничения на веса
        cons = lambda w: np.sum(w) - 1
        bounds = [(0.0, 1.0)] * n_features
        
        # Оптимизация весов
        self.weights_ = fmin_slsqp(
            partial(self.loss, X=self.X_, y=self.y_),
            init_w,
            f_eqcons=cons,
            bounds=bounds,
            disp=False
        )
        
    def predict(self) -> np.ndarray:
        """
        Предсказание значений для обработанных единиц.
        
        Returns
        -------
        np.ndarray
            Предсказанные значения
        """
        if self.weights_ is None:
            raise ValueError("Модель не обучена. Вызовите метод fit() перед predict()")
        
        # Получение данных для контрольных единиц
        control_data = self.data[~self.data[self.treated]]
        
        # Преобразование в матричный формат
        x_all_control = control_data.pivot(
            index=self.period_index,
            columns=self.shopno,
            values=self.metric
        )
        
        # Проверка, что все контрольные единицы присутствуют
        missing_units = set(self.control_units_) - set(x_all_control.columns)
        if missing_units:
            raise ValueError(f"Отсутствуют данные для контрольных единиц: {missing_units}")
        
        # Упорядочивание колонок в соответствии с control_units_
        x_all_control = x_all_control[self.control_units_]
        
        # Предсказание
        return x_all_control.values @ self.weights_
        
    def estimate_effect(self) -> Dict[str, float]:
        """
        Оценка эффекта вмешательства.
        
        Returns
        -------
        Dict[str, float]
            Словарь с оценкой эффекта и весами контрольных единиц
        """
        if self.weights_ is None:
            raise ValueError("Модель не обучена. Вызовите метод fit() перед estimate_effect()")
        
        # Получение предсказаний для всего периода
        y_pred = self.predict()
        
        # Получение данных для обработанной единицы в пост-интервенционном периоде
        y_post_treat = (self.data
            .query(f"{self.treated} and {self.after_treatment}")
            .sort_values(self.period_index)
            [self.metric]
            .values
        )
        
        if len(y_post_treat) == 0:
            raise ValueError("Нет данных для обработанной единицы в пост-интервенционном периоде")
        
        # Получение предсказаний для пост-интервенционного периода
        sc_post = y_pred[-len(y_post_treat):]
        
        # Проверка размерностей
        if len(y_post_treat) != len(sc_post):
            raise ValueError(
                f"Несоответствие размерностей: len(y_post_treat)={len(y_post_treat)}, "
                f"len(sc_post)={len(sc_post)}"
            )
        
        # Расчет эффекта
        att = np.mean(y_post_treat - sc_post)
        
        return {
            'att': att,
            'weights': dict(zip(self.control_units_, self.weights_))
        }
        
    def bootstrap_effect(self, alpha: float = 0.05, ci_method: str = 'normal') -> Dict[str, float]:
        """
        Оценка стандартной ошибки эффекта с помощью бутстрепа.

        Parameters
        ----------
        alpha : float, default=0.05
            Уровень значимости для доверительного интервала (1-alpha)
        ci_method : str, default='normal'
            Способ вычисления доверительного интервала: 'normal', 'percentile', 't'

        Returns
        -------
        Dict[str, float]
            Словарь со стандартной ошибкой и границами доверительного интервала
        """
        if self.weights_ is None:
            raise ValueError("Модель не обучена. Вызовите метод fit() перед bootstrap_effect()")

        np.random.seed(self.seed)
        effects = []
        att = self.estimate_effect()['att']

        for _ in range(self.bootstrap_rounds):
            control = self.data[~self.data[self.treated]]
            shopnos = control[self.shopno].unique()
            placebo_shopno = np.random.choice(shopnos)
            placebo_data = control.assign(
                **{self.treated: control[self.shopno] == placebo_shopno}
            )
            placebo_model = ClassicSyntheticControl(
                data=placebo_data,
                metric=self.metric,
                period_index=self.period_index,
                shopno=self.shopno,
                treated=self.treated,
                after_treatment=self.after_treatment
            )
            placebo_model.fit()
            effect = placebo_model.estimate_effect()['att']
            effects.append(effect)

        se = np.std(effects, ddof=1)
        n = len(effects)

        if ci_method == 'normal':
            z = norm.ppf(1 - alpha / 2)
            ci_lower = att - z * se
            ci_upper = att + z * se
        elif ci_method == 'percentile':
            ci_lower = np.percentile(effects, 100 * alpha / 2)
            ci_upper = np.percentile(effects, 100 * (1 - alpha / 2))
        elif ci_method == 't':
            t_crit = t.ppf(1 - alpha / 2, df=n - 1)
            ci_lower = att - t_crit * se
            ci_upper = att + t_crit * se
        else:
            raise ValueError("ci_method должен быть 'normal', 'percentile' или 't'")

        return {
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }