import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy.optimize import fmin_slsqp
from functools import partial
from .base import SyntheticControl
from scipy.stats import norm
from scipy.stats import norm, t
import statsmodels.formula.api as smf
from joblib import Parallel, delayed

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

class SyntheticDIDModel(SyntheticControl):
    """
    Реализация метода Synthetic Difference-in-Differences.
    
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
    seed : int, default=42
        Seed для воспроизводимости результатов
    bootstrap_rounds : int, default=100
        Количество раундов бутстрепа для оценки стандартной ошибки
    njobs : int, default=4
        Количество параллельных задач для вычисления стандартной ошибки
    """
    def __init__(self, data, metric, period_index, shopno, treated, after_treatment,
                 seed=42, bootstrap_rounds=100, njobs=4):
        # Вызываем конструктор родительского класса
        super().__init__(
            data=data,
            metric=metric,
            period_index=period_index,
            shopno=shopno,
            treated=treated,
            after_treatment=after_treatment,
            bootstrap_rounds=bootstrap_rounds,
            seed=seed
        )
        # Переопределяем названия колонок для совместимости
        self.outcome_col = metric
        self.period_index_col = period_index
        self.shopno_col = shopno
        self.treat_col = treated
        self.post_col = after_treatment
        self.njobs = njobs

    def loss(self, w, X, y):
        """
        Функция потерь для оптимизации весов времени.
        
        Parameters
        ----------
        w : np.ndarray
            Вектор весов
        X : np.ndarray
            Матрица признаков
        y : np.ndarray
            Вектор целевых значений
            
        Returns
        -------
        float
            Значение функции потерь
        """
        pred = X.T.dot(w)
        return np.sqrt(np.mean((y - pred)**2))
    
    def loss_penalized(self, w, X, y, T_pre, zeta):
        """
        Штрафная функция потерь для оптимизации весов единиц.
        
        Parameters
        ----------
        w : np.ndarray
            Вектор весов
        X : np.ndarray
            Матрица признаков
        y : np.ndarray
            Вектор целевых значений
        T_pre : int
            Количество предшествующих периодов
        zeta : float
            Параметр регуляризации
            
        Returns
        -------
        float
            Значение функции потерь
        """
        resid = X.dot(w) - y
        return np.sum(resid**2) + T_pre * (zeta**2) * np.sum(w[1:]**2)

    def calculate_regularization(self, data):
        """
        Расчет параметра регуляризации.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame с данными
            
        Returns
        -------
        float
            Значение параметра регуляризации
        """
        if self.post_col not in data.columns or self.treat_col not in data.columns:
            raise ValueError(f"Отсутствуют необходимые столбцы: {self.post_col} или {self.treat_col}")
            
        n_treated_post = data.loc[(data[self.post_col] == 1) & (data[self.treat_col] == 1)].shape[0]
        first_diff_std = (data
                          .loc[(data[self.post_col] == 0) & (data[self.treat_col] == 0)]
                          .sort_values(self.period_index_col)
                          .groupby(self.shopno_col)[self.outcome_col]
                          .diff()
                          .std())
        return n_treated_post ** (1 / 4) * first_diff_std

    def join_weights(self, data, unit_w, time_w):
        """
        Объединение весов времени и единиц в одну таблицу.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame с данными
        unit_w : pd.Series
            Веса единиц
        time_w : pd.Series
            Веса времени
            
        Returns
        -------
        pd.DataFrame
            DataFrame с объединенными весами
        """
        joined = (data
                  .set_index([self.period_index_col, self.shopno_col])
                  .join(time_w)
                  .join(unit_w)
                  .reset_index()
                  .fillna({
                      time_w.name: 1 / len(pd.unique(data.loc[data[self.post_col] == 1, self.period_index_col])),
                      unit_w.name: 1 / len(pd.unique(data.loc[data[self.treat_col] == 1, self.shopno_col]))
                  })
                  .assign(**{"weights": lambda d: (d[time_w.name] * d[unit_w.name]).round(10)})
                  .astype({self.treat_col: int, self.post_col: int}))
        return joined

    def fit_time_weights(self, data):
        """
        Вычисление весов времени.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame с данными
            
        Returns
        -------
        pd.Series
            Серия с весами времени
        """
        control = data.loc[data[self.treat_col] == 0]
        y_pre = (control
                 .loc[control[self.post_col] == 0]
                 .pivot(index=self.period_index_col, columns=self.shopno_col, values=self.outcome_col))
        y_post_mean = (control
                       .loc[control[self.post_col] == 1]
                       .groupby(self.shopno_col)[self.outcome_col]
                       .mean()
                       .values)

        X = np.vstack([np.ones((1, y_pre.shape[1])), y_pre.values])
        n_features, n_shops = X.shape
        init_w = np.ones(n_features) / n_features

        cons = lambda w, *args: np.sum(w[1:]) - 1

        bounds = [(None, None)] + [(0.0, 1.0)] * (n_features - 1)

        opt_w = fmin_slsqp(
            func=partial(self.loss, X=X, y=y_post_mean),
            x0=init_w,
            f_eqcons=cons,
            bounds=bounds,
            disp=False
        )

        return pd.Series(opt_w[1:], name="time_weights", index=y_pre.index)

    def fit_unit_weights(self, data):
        """
        Вычисление весов единиц.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame с данными
            
        Returns
        -------
        tuple
            (pd.Series с весами единиц, float константа)
        """
        zeta = self.calculate_regularization(data)
        pre_data = data.loc[data[self.post_col] == 0]
        y_pre_control = (pre_data
                         .loc[pre_data[self.treat_col] == 0]
                         .pivot(index=self.period_index_col, columns=self.shopno_col, values=self.outcome_col))
        y_pre_treat_mean = (pre_data
                            .loc[pre_data[self.treat_col] == 1]
                            .groupby(self.period_index_col)[self.outcome_col]
                            .mean())
        T_pre = y_pre_control.shape[0]
        
        X = np.concatenate([np.ones((T_pre, 1)), y_pre_control.values], axis=1)
        
        cons = lambda w, *args: np.sum(w[1:]) - 1

        n_coef = X.shape[1]
        init_w = np.ones(n_coef) / n_coef

        bounds = [(None, None)] + [(0.0, 1.0)] * (n_coef - 1)
        
        opt_w = fmin_slsqp(
            func = partial(self.loss_penalized, X=X, y=y_pre_treat_mean.values, T_pre=T_pre, zeta=zeta),
            x0 = init_w,
            f_eqcons = cons,
            bounds = bounds,
            disp = False
        )
        return pd.Series(opt_w[1:], name="unit_weights", index=y_pre_control.columns), opt_w[0]

    def synthetic_diff_in_diff(self, data=None):
        """
        Вычисление эффекта методом Synthetic Difference-in-Differences.
        
        Parameters
        ----------
        data : pd.DataFrame, optional
            DataFrame с данными, по умолчанию None
            
        Returns
        -------
        tuple
            (float эффект, pd.Series веса единиц, pd.Series веса времени, 
             statsmodels.regression.linear_model.RegressionResultsWrapper модель, float константа)
        """
        if data is None:
            data = self.data
        unit_weights, intercept = self.fit_unit_weights(data)
        time_weights = self.fit_time_weights(data)
        did_data = self.join_weights(data, unit_weights, time_weights)
        formula = f"{self.outcome_col} ~ {self.post_col}*{self.treat_col}"
        did_model = smf.wls(formula, data=did_data, weights=did_data["weights"] + 1e-10).fit()
        att = did_model.params[f"{self.post_col}:{self.treat_col}"]
        return att, unit_weights, time_weights, did_model, intercept

    def make_random_placebo(self, data):
        """
        Создание плацебо данных для бутстрапа.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame с данными
            
        Returns
        -------
        pd.DataFrame
            DataFrame с плацебо данными
        """
        control = data.query(f"~{self.treat_col}")
        shopnos = control[self.shopno_col].unique()
        placebo_shopno = np.random.choice(shopnos)
        return control.assign(**{self.treat_col: control[self.shopno_col] == placebo_shopno})

    def _single_placebo_att(self, seed):
        """
        Вычисление одного плацебо эффекта.
        
        Parameters
        ----------
        seed : int
            Seed для воспроизводимости
            
        Returns
        -------
        float
            Значение плацебо эффекта
        """
        np.random.seed(seed)
        placebo_data = self.make_random_placebo(self.data)
        att_placebo, *_ = self.synthetic_diff_in_diff(data=placebo_data)
        return att_placebo

    def estimate_se(self, alpha=0.05):
        """
        Оценка стандартной ошибки и доверительного интервала.
        
        Parameters
        ----------
        alpha : float, default=0.05
            Уровень значимости
            
        Returns
        -------
        tuple
            (float эффект, float стандартная ошибка, float нижняя граница ДИ, float верхняя граница ДИ)
        """
        master_rng = np.random.RandomState(self.seed)
        main_att, *_ = self.synthetic_diff_in_diff()

        seeds = master_rng.randint(low=0, high=2**31-1,
                                   size=self.bootstrap_rounds)

        effects = Parallel(n_jobs=self.njobs)(
            delayed(self._single_placebo_att)(seed)
            for seed in seeds
        )

        se = np.std(effects, ddof=1)
        z  = norm.ppf(1 - alpha/2)
        return main_att, se, main_att - z*se, main_att + z*se

    def fit(self):
        """
        Обучение модели.
        
        Returns
        -------
        None
        """
        self.att_, self.unit_weights_, self.time_weights_, self.model_, self.intercept_ = self.synthetic_diff_in_diff()
        
    def __repr__(self):
        """
        Строковое представление объекта для отладки.
        
        Returns
        -------
        str
            Строковое представление
        """
        if hasattr(self, 'att_'):
            return f"SyntheticDIDModel(ATT={self.att_:.4f})"
        else:
            return "SyntheticDIDModel(not fitted)"
    
    def __str__(self):
        """
        Строковое представление объекта для пользователя.
        
        Returns
        -------
        str
            Строковое представление
        """
        if hasattr(self, 'att_'):
            return f"Модель Synthetic DID с эффектом ATT = {self.att_:.4f}"
        else:
            return "Модель Synthetic DID (не обучена)"
        
    def predict(self):
        """
        Предсказание значений для обработанной группы в период после вмешательства.
        
        Returns
        -------
        np.ndarray
            Массив с предсказанными значениями
        """
        if not hasattr(self, 'model_'):
            raise ValueError("Необходимо сначала обучить модель с помощью метода fit()")
            
        treated_post = self.data.query(f"{self.treat_col} and {self.post_col}")
        treated_post = self.join_weights(treated_post, self.unit_weights_, self.time_weights_)
        
        # Создаем копию данных, где treated = 0 для получения контрфактического результата
        counterfactual = treated_post.copy()
        counterfactual[self.treat_col] = 0
        
        # Предсказываем с моделью
        return self.model_.predict(counterfactual)
        
    def estimate_effect(self):
        """
        Оценка эффекта вмешательства.
        
        Returns
        -------
        dict
            Словарь с оценками эффекта и весами контрольных единиц
        """
        if not hasattr(self, 'att_'):
            self.fit()
            
        # Создаем словарь весов
        weights_dict = self.unit_weights_.to_dict()
        
        return {
            'att': self.att_,
            'weights': weights_dict
        }

    def bootstrap_effect(self, alpha=0.05):
        """
        Оценка стандартной ошибки эффекта с помощью бутстрепа.
        
        Parameters
        ----------
        alpha : float, default=0.05
            Уровень значимости для доверительного интервала
            
        Returns
        -------
        dict
            Словарь со стандартной ошибкой и границами доверительного интервала
        """
        if not hasattr(self, 'att_'):
            self.fit()
            
        _, se, ci_lower, ci_upper = self.estimate_se(alpha=alpha)
        
        return {
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }