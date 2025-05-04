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
    unit_id : str
        Название колонки с идентификаторами единиц
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
        df_pre = self.data[~self.data[self.after_treatment]]
        
        treated_data = df_pre[df_pre[self.treated]]
        if len(treated_data) == 0:
            raise ValueError("Нет данных для обработанной единицы в предварительном периоде")
        
        control_data = df_pre[~df_pre[self.treated]]
        if len(control_data) == 0:
            raise ValueError("Нет данных для контрольных единиц")
        
        df_pre_control = control_data.pivot(
            index=self.period_index,
            columns=self.unit_id,
            values=self.metric
        )
        
        self.control_units_ = list(df_pre_control.columns)
        self.X_ = df_pre_control.values
        
        self.y_ = treated_data.groupby(self.period_index)[self.metric].mean().values
        
        if self.X_.shape[0] != len(self.y_):
            raise ValueError(
                f"Несоответствие размерностей: X_.shape[0]={self.X_.shape[0]}, "
                f"len(y_)={len(self.y_)}"
            )
        
        n_features = self.X_.shape[1]
        init_w = np.ones(n_features) / n_features
        
        cons = lambda w: np.sum(w) - 1
        bounds = [(0.0, 1.0)] * n_features
        
        weights_array = fmin_slsqp(
            partial(self.loss, X=self.X_, y=self.y_),
            init_w,
            f_eqcons=cons,
            bounds=bounds,
            disp=False
        )
        
        self.weights_ = pd.Series(weights_array, index=self.control_units_, name="weights")
        
    def predict(self) -> np.ndarray:
        """
        Предсказание значений для обработанных единиц.
        
        Returns
        -------
        np.ndarray
            Предсказанные значения для всех периодов обработанной единицы
        """
        if self.weights_ is None:
            raise ValueError("Модель не обучена. Вызовите метод fit() перед predict()")
        
        control_data = self.data[~self.data[self.treated]]
        
        treated_data = self.data[self.data[self.treated]].sort_values(self.period_index)
        treated_periods = sorted(treated_data[self.period_index].unique())
        
        x_all_control = control_data.pivot(
            index=self.period_index,
            columns=self.unit_id,
            values=self.metric
        )
        
        missing_units = set(self.control_units_) - set(x_all_control.columns)
        if missing_units:
            raise ValueError(f"Отсутствуют данные для контрольных единиц: {missing_units}")
        
        x_all_control = x_all_control[self.control_units_]
        
        predictions_control = x_all_control.values @ self.weights_.values
        
        all_predictions = np.full(len(treated_periods), np.nan)
        
        for i, period in enumerate(treated_periods):
            if period in x_all_control.index:
                period_index = x_all_control.index.get_loc(period)
                all_predictions[i] = predictions_control[period_index]
        
        if np.all(np.isnan(all_predictions)):
            raise ValueError("Не удалось получить предсказания для периодов обработанной группы")
        
        return all_predictions
        
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
        
        y_pred = self.predict()
        
        treated_data = self.data[self.data[self.treated]].sort_values(self.period_index)
        treated_periods = treated_data[self.period_index].unique()
        
        y_post_treat = treated_data[treated_data[self.after_treatment]][self.metric].values
        post_periods = treated_data[treated_data[self.after_treatment]][self.period_index].values
        
        if len(y_post_treat) == 0:
            raise ValueError("Нет данных для обработанной единицы в пост-интервенционном периоде")
        
        post_indices = [i for i, p in enumerate(treated_periods) if p in post_periods]
        
        sc_post = y_pred[post_indices]
        
        valid_indices = ~np.isnan(sc_post)
        if not np.any(valid_indices):
            raise ValueError("Нет валидных предсказаний для пост-интервенционного периода")
        
        valid_y_post = y_post_treat[valid_indices]
        valid_sc_post = sc_post[valid_indices]
        
        att = np.mean(valid_y_post - valid_sc_post)
        
        return {
            'att': att,
            'weights': self.weights_
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
            shopnos = control[self.unit_id].unique()
            placebo_shopno = np.random.choice(shopnos)
            placebo_data = control.assign(
                **{self.treated: control[self.unit_id] == placebo_shopno}
            )
            placebo_model = ClassicSyntheticControl(
                data=placebo_data,
                metric=self.metric,
                period_index=self.period_index,
                unit_id=self.unit_id,
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
        
    def plot_model_results(self, T0=None, figsize=(14, 7), save_path=None, show=False):
        """
        Визуализация результатов модели Synthetic Control.
        
        Parameters
        ----------
        T0 : int или float, optional
            Период начала воздействия. Если None, берется из модели.
        figsize : tuple, default=(14, 7)
            Размер графика
        save_path : str, optional
            Путь для сохранения графика
        show : bool, default=False
            Отображать ли график автоматически
            
        Returns
        -------
        matplotlib.figure.Figure
            Объект фигуры matplotlib
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        
        plt.close('all')
        
        data = self.data.copy()
        outcome_col = self.metric
        period_index_col = self.period_index
        treat_col = self.treated
        post_col = self.after_treatment
        
        if T0 is None:
            if hasattr(self, 'treatment_date') and self.treatment_date is not None:
                T0 = self.treatment_date
            else:
                T0 = data[data[post_col]].sort_values(period_index_col)[period_index_col].min()
                print(f"Warning: Treatment date not specified, using first post-treatment period: {T0}")
        
        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=figsize)
        
        treated_data = data[data[treat_col]].sort_values(period_index_col)
        treated_values = treated_data.groupby(period_index_col)[outcome_col].mean()
        
        ax.plot(treated_values.index, treated_values.values, 
                label="Treatment Group", color="red", linewidth=2)
        
        try:
            predictions = self.predict()
            
            if len(predictions) < len(treated_values):
                
                control_periods = data[~data[treat_col]].sort_values(period_index_col)[period_index_col].unique()
                control_periods = [p for p in control_periods if p in treated_values.index]
                
                if len(control_periods) == len(predictions):
                    pred_df = pd.DataFrame({
                        'period': control_periods,
                        'predicted': predictions
                    }).set_index('period')
                    
                    full_predictions = np.full(len(treated_values), np.nan)
                    for i, period in enumerate(treated_values.index):
                        if period in pred_df.index:
                            full_predictions[i] = pred_df.loc[period, 'predicted']
                    
                    predictions = full_predictions
                    if np.isnan(predictions).any():
                        print("Warning: Some periods have no predictions.")
            
            elif len(predictions) > len(treated_values):
                all_periods = sorted(data[period_index_col].unique())
                
                if len(all_periods) == len(predictions):
                    pred_df = pd.DataFrame({
                        'period': all_periods,
                        'predicted': predictions
                    }).set_index('period')
                    
                    predictions = np.array([pred_df.loc[p, 'predicted'] if p in pred_df.index else np.nan 
                                          for p in treated_values.index])
                else:
                    predictions = predictions[-len(treated_values):]
            
            ax.plot(treated_values.index, predictions, 
                    label="Synthetic Control", color="blue", linestyle="--", linewidth=2)
            
            ax.axvline(x=T0, color='black', linestyle=':', label='Treatment Start')
            
            post_periods = [p for p in treated_values.index if p >= T0]
            if post_periods:
                post_treated = treated_values.loc[post_periods]
                post_pred_indices = [i for i, p in enumerate(treated_values.index) if p in post_periods]
                post_predicted = predictions[post_pred_indices]
                
                valid_indices = ~np.isnan(post_predicted)
                if np.any(valid_indices):
                    att = np.mean(np.array(post_treated)[valid_indices] - post_predicted[valid_indices])
                    
                    ax.text(0.05, 0.95, 
                          f"ATT: {att:.4f}",
                          transform=ax.transAxes, fontsize=12, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    for i, period in enumerate(post_periods):
                        if i < len(post_predicted) and not np.isnan(post_predicted[i]):
                            actual = post_treated.iloc[i]
                            pred = post_predicted[i]
                            ax.plot([period, period], [pred, actual], 'r-', alpha=0.7)
        
        except Exception as e:
            print(f"Error building predictions plot: {str(e)}")
        
        ax.set_title("Synthetic Control Results", fontsize=14)
        ax.set_xlabel(period_index_col, fontsize=12)
        ax.set_ylabel(outcome_col, fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
            
        return fig
    
    def plot_cumulative_effect(self, treatment_date=None, figsize=(12, 6), title=None, 
                             xlabel=None, ylabel=None, show=False):
        """
        Визуализация кумулятивного эффекта между фактическими и предсказанными значениями.
        
        Parameters
        ----------
        treatment_date : int, optional
            Дата воздействия. Если None, берется из модели.
        figsize : tuple, default=(12, 6)
            Размер графика
        title : str, optional
            Заголовок графика
        xlabel : str, optional
            Подпись оси X
        ylabel : str, optional
            Подпись оси Y
        show : bool, default=False
            Отображать ли график автоматически
            
        Returns
        -------
        matplotlib.figure.Figure
            Объект фигуры matplotlib
        """
        from .visualization import plot_cumulative_effect
        
        if not hasattr(self, 'weights_') or self.weights_ is None:
            raise ValueError("Модель должна быть обучена перед визуализацией. Вызовите метод fit().")
        
        predictions = self.predict()
        
        if treatment_date is None:
            treatment_date = self.treatment_date
        
        if title is None:
            title = f"Cumulative Effect of Synthetic Control"
            
        if xlabel is None:
            xlabel = self.period_index
        if ylabel is None:
            ylabel = "Cumulative Difference"
        
        return plot_cumulative_effect(
            data=self.data,
            metric=self.metric,
            period_index=self.period_index,
            treated=self.treated,
            predictions=predictions,
            treatment_date=treatment_date,
            figsize=figsize,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            show=show
        )

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
    unit_id : str
        Название колонки с идентификаторами единиц
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
    def __init__(self, data, metric, period_index, unit_id, treated, after_treatment,
                 seed=42, bootstrap_rounds=100, njobs=4):
        super().__init__(
            data=data,
            metric=metric,
            period_index=period_index,
            unit_id=unit_id,
            treated=treated,
            after_treatment=after_treatment,
            bootstrap_rounds=bootstrap_rounds,
            seed=seed
        )
        self.outcome_col = metric
        self.period_index_col = period_index
        self.shopno_col = unit_id
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
            Массив с предсказанными значениями для всех периодов обработанной единицы
        """
        if not hasattr(self, 'model_'):
            raise ValueError("Необходимо сначала обучить модель с помощью метода fit()")
        
        treated_data = self.data[self.data[self.treat_col]].sort_values(self.period_index_col)
        treated_periods = sorted(treated_data[self.period_index_col].unique())
        
        all_predictions = np.full(len(treated_periods), np.nan)
        
        treated_post = self.data.query(f"{self.treat_col} and {self.post_col}")
        if not treated_post.empty:
            treated_post = self.join_weights(treated_post, self.unit_weights_, self.time_weights_)
            
            counterfactual = treated_post.copy()
            counterfactual[self.treat_col] = 0
            
            post_predictions = self.model_.predict(counterfactual)
            
            post_periods = treated_post[self.period_index_col].values
            for i, period in enumerate(treated_periods):
                if period in post_periods:
                    idx = np.where(post_periods == period)[0][0]
                    all_predictions[i] = post_predictions[idx]
        
        treated_pre = self.data.query(f"{self.treat_col} and not {self.post_col}")
        if not treated_pre.empty:
            pre_periods = treated_pre[self.period_index_col].values
            pre_values = treated_pre[self.outcome_col].values
            
            for i, period in enumerate(treated_periods):
                if period in pre_periods:
                    idx = np.where(pre_periods == period)[0][0]
                    all_predictions[i] = pre_values[idx]
        
        if np.all(np.isnan(all_predictions)):
            raise ValueError("Не удалось получить предсказания для периодов обработанной группы")
        
        return all_predictions
        
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
    
    def plot_model_results(self, T0=None, figsize=(14, 7), save_path=None, show=False):
        """
        Визуализация результатов модели Synthetic DID.
        
        Parameters
        ----------
        T0 : int или float, optional
            Период начала воздействия. Если None, берется из модели.
        figsize : tuple, default=(14, 7)
            Размер графика
        save_path : str, optional
            Путь для сохранения графика
        show : bool, default=False
            Отображать ли график автоматически
            
        Returns
        -------
        matplotlib.figure.Figure
            Объект фигуры matplotlib
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        
        plt.close('all')
        
        if not hasattr(self, 'model_'):
            raise ValueError("Model must be trained before visualization. Call fit() method")
        
        if T0 is None:
            if hasattr(self, 'treatment_date') and self.treatment_date is not None:
                T0 = self.treatment_date
            else:
                T0 = self.data[self.data[self.post_col]].sort_values(self.period_index_col)[self.period_index_col].min()
                print(f"Warning: Treatment date not specified, using first post-treatment period: {T0}")
        
        try:
            att, unit_weights, time_weights, sdid_model_fit, intercept = self.synthetic_diff_in_diff()
            
            y_co_all = self.data.loc[self.data[self.treat_col] == 0] \
                          .pivot(index=self.period_index_col, columns=self.shopno_col, values=self.outcome_col) \
                          .sort_index()
            sc_did = intercept + y_co_all.dot(unit_weights)
            
            treated_all = self.data.loc[self.data[self.treat_col] == 1] \
                              .groupby(self.period_index_col)[self.outcome_col].mean()
            
            pre_times = self.data.loc[self.data[self.period_index_col] < T0, self.period_index_col]
            post_times = self.data.loc[self.data[self.period_index_col] >= T0, self.period_index_col]
            avg_pre_period = pre_times.mean() if len(pre_times) > 0 else T0
            avg_post_period = post_times.mean() if len(post_times) > 0 else T0 + 1
            
            params = sdid_model_fit.params
            pre_sc = params.get("Intercept", 0)
            post_sc = pre_sc + params.get(self.post_col, 0)
            pre_treat = pre_sc + params.get(self.treat_col, 0)
            
            post_treat_key = f"{self.post_col}:{self.treat_col}"
            if post_treat_key in params:
                post_treat = post_sc + params[self.treat_col] + params[post_treat_key]
            else:
                post_treat = pre_treat
            
            sc_did_y0 = pre_treat + (post_sc - pre_sc)
            
            plt.style.use("ggplot")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                          gridspec_kw={'height_ratios': [3, 1]})
            
            ax1.plot(sc_did.index, sc_did.values, label="Synthetic DID", color="black", alpha=0.8)
            ax1.plot(treated_all.index, treated_all.values, label="Treatment Group", color="red", linewidth=2)
            
            ax1.plot([avg_pre_period, avg_post_period], [pre_sc, post_sc],
                    color="#1f77b4", label="Counterfactual Trend", linewidth=2)
            ax1.plot([avg_pre_period, avg_post_period], [pre_treat, post_treat],
                    color="#ff7f0e", linestyle="dashed", label="Effect", linewidth=2)
            ax1.plot([avg_pre_period, avg_post_period], [pre_treat, sc_did_y0],
                    color="#ff7f0e", label="Synthetic Trend", linewidth=2)
            
            x_bracket = avg_post_period
            y_top = post_treat
            y_bottom = sc_did_y0
            ax1.annotate(
                '', 
                xy=(x_bracket, y_bottom), 
                xytext=(x_bracket, y_top),
                arrowprops=dict(arrowstyle='|-|', color='#9467bd', lw=2)
            )
            
            ax1.text(x_bracket + 0.5, (y_top + y_bottom) / 2, f"ATT = {round(att, 4)}",
                    color='black', fontsize=12, va='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax1.legend()
            ax1.set_title("Synthetic Difference-in-Differences")
            ax1.axvline(T0, color='black', linestyle=':', label='Treatment Start')
            ax1.set_ylabel(self.outcome_col)

            ax2.bar(time_weights.index, time_weights.values, color='blue', alpha=0.7)
            ax2.axvline(T0, color="black", linestyle="dotted")
            ax2.set_ylabel("Time Weights")
            ax2.set_xlabel(self.period_index_col)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            if show:
                plt.show()
                
            return fig
        
        except Exception as e:
            print(f"Error building Synthetic DID plot: {str(e)}")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"Error building plot: {str(e)}", 
                  ha='center', va='center', transform=ax.transAxes)
            if show:
                plt.show()
            return fig
    
    def plot_cumulative_effect(self, treatment_date=None, figsize=(12, 6), title=None, 
                             xlabel=None, ylabel=None, show=False):
        """
        Визуализация кумулятивного эффекта между фактическими и предсказанными значениями.
        
        Parameters
        ----------
        treatment_date : int, optional
            Дата воздействия. Если None, берется из модели.
        figsize : tuple, default=(12, 6)
            Размер графика
        title : str, optional
            Заголовок графика
        xlabel : str, optional
            Подпись оси X
        ylabel : str, optional
            Подпись оси Y
        show : bool, default=False
            Отображать ли график автоматически
            
        Returns
        -------
        matplotlib.figure.Figure
            Объект фигуры matplotlib
        """
        from .visualization import plot_cumulative_effect
        
        if not hasattr(self, 'model_'):
            raise ValueError("Модель должна быть обучена перед визуализацией. Вызовите метод fit().")
        
        predictions = self.predict()
        
        if treatment_date is None:
            treatment_date = self.treatment_date
        
        if title is None:
            title = f"Cumulative Effect of Synthetic DID"
            
        if xlabel is None:
            xlabel = self.period_index_col
        if ylabel is None:
            ylabel = "Cumulative Difference"
        
        return plot_cumulative_effect(
            data=self.data,
            metric=self.outcome_col,
            period_index=self.period_index_col,
            treated=self.treat_col,
            predictions=predictions,
            treatment_date=treatment_date,
            figsize=figsize,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            show=show
        )