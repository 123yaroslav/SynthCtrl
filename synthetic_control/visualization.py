import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional
import seaborn as sns

def plot_synthetic_control(
    data: pd.DataFrame,
    metric: str,
    period_index: str,
    shopno: str,
    treated: str,
    after_treatment: str,
    predictions: np.ndarray,
    treatment_date: Optional[int] = None,
    figsize: tuple = (12, 6),
    title: str = "Synthetic Control",
    xlabel: str = "Date",
    ylabel: str = "Metric",
    show: bool = False
) -> plt.Figure:
    """
    Визуализация результатов Synthetic Control.
    
    Parameters
    ----------
    data : pd.DataFrame
        Исходные данные
    metric : str
        Название метрики
    period_index : str
        Название колонки с периодами
    shopno : str
        Название колонки с идентификаторами магазинов
    treated : str
        Название колонки, указывающей на обработанные единицы
    after_treatment : str
        Название колонки, указывающей на периоды после вмешательства
    predictions : np.ndarray
        Предсказанные значения
    treatment_date : Optional[int]
        Дата вмешательства
    figsize : tuple, default=(12, 6)
        Размер графика
    title : str, default="Synthetic Control"
        Заголовок графика
    xlabel : str, default="Date"
        Подпись оси X
    ylabel : str, default="Metric"
        Подпись оси Y
    show : bool, default=False
        Отображать ли график автоматически
        
    Returns
    -------
    plt.Figure
        Объект графика
    """
    # Только обработанная единица (например, Калифорния)
    treated_data = data[data[treated]].sort_values(period_index)
    periods = treated_data[period_index].values
    actual = treated_data[metric].values

    # predictions должен быть той же длины, что и actual
    if len(predictions) != len(actual):
        # Если predictions длиннее, берем только последние значения
        if len(predictions) > len(actual):
            predictions = predictions[-len(actual):]
        else:
            raise ValueError("Длина predictions меньше длины фактических данных для обработанной единицы.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(periods, actual, label='Treated', color='blue')
    ax.plot(periods, predictions, label='Synthetic Control', color='red', linestyle='--')

    if treatment_date is not None:
        ax.axvline(x=treatment_date, color='black', linestyle=':', label='Treatment')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    
    if show:
        plt.show()
        
    return fig

def plot_effect_distribution(
    effects: np.ndarray,
    observed_effect: float,
    figsize: tuple = (10, 6),
    title: str = "Effect Distribution",
    xlabel: str = "Effect",
    ylabel: str = "Density",
    show: bool = False
) -> plt.Figure:
    """
    Визуализация распределения эффектов.
    
    Parameters
    ----------
    effects : np.ndarray
        Массив эффектов
    observed_effect : float
        Наблюдаемый эффект
    figsize : tuple, default=(10, 6)
        Размер графика
    title : str, default="Effect Distribution"
        Заголовок графика
    xlabel : str, default="Effect"
        Подпись оси X
    ylabel : str, default="Density"
        Подпись оси Y
    show : bool, default=False
        Отображать ли график автоматически
        
    Returns
    -------
    plt.Figure
        Объект графика
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Построение гистограммы
    sns.histplot(effects, kde=True, ax=ax)
    
    # Добавление вертикальной линии для наблюдаемого эффекта
    ax.axvline(x=observed_effect, color='red', linestyle='--', label='Observed Effect')
    
    # Настройка графика
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    
    if show:
        plt.show()
    
    return fig

def plot_weights(
    weights: Dict[str, float],
    figsize: tuple = (10, 6),
    title: str = "Control Unit Weights",
    xlabel: str = "Control Unit",
    ylabel: str = "Weight",
    show: bool = False
) -> plt.Figure:
    """
    Визуализация весов контрольных единиц.
    
    Parameters
    ----------
    weights : Dict[str, float]
        Словарь с весами контрольных единиц
    figsize : tuple, default=(10, 6)
        Размер графика
    title : str, default="Control Unit Weights"
        Заголовок графика
    xlabel : str, default="Control Unit"
        Подпись оси X
    ylabel : str, default="Weight"
        Подпись оси Y
    show : bool, default=False
        Отображать ли график автоматически
        
    Returns
    -------
    plt.Figure
        Объект графика
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Подготовка данных
    units = list(weights.keys())
    values = list(weights.values())
    
    # Построение столбчатой диаграммы
    ax.bar(units, values)
    
    # Настройка графика
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    
    # Поворот подписей на оси X
    plt.xticks(rotation=45)
    
    if show:
        plt.show()
    
    return fig 

def plot_cumulative_effect(
    data: pd.DataFrame,
    metric: str,
    period_index: str,
    treated: str,
    predictions: np.ndarray,
    treatment_date: Optional[int] = None,
    figsize: tuple = (12, 6),
    title: str = "Кумулятивный эффект Synthetic Control",
    xlabel: str = "Date",
    ylabel: str = "Кумулятивная разница",
    show: bool = False
) -> plt.Figure:
    """
    Визуализация кумулятивного эффекта между Treated и Synthetic Control.
    
    Parameters
    ----------
    data : pd.DataFrame
        Исходные данные
    metric : str
        Название метрики
    period_index : str
        Название колонки с периодами
    treated : str
        Название колонки, указывающей на обработанные единицы
    predictions : np.ndarray
        Предсказанные значения (Synthetic Control)
    treatment_date : Optional[int]
        Дата вмешательства
    figsize : tuple, default=(12, 6)
        Размер графика
    title : str, default="Кумулятивный эффект Synthetic Control"
        Заголовок графика
    xlabel : str, default="Date"
        Подпись оси X
    ylabel : str, default="Кумулятивная разница"
        Подпись оси Y
    show : bool, default=False
        Отображать ли график автоматически
    
    Returns
    -------
    plt.Figure
        Объект графика
    """
    # Только обработанная единица (например, Калифорния)
    treated_data = data[data[treated]].sort_values(period_index)
    periods = treated_data[period_index].values
    actual = treated_data[metric].values

    # predictions должен быть той же длины, что и actual
    if len(predictions) != len(actual):
        if len(predictions) > len(actual):
            predictions = predictions[-len(actual):]
        else:
            raise ValueError("Длина predictions меньше длины фактических данных для обработанной единицы.")

    # Кумулятивная сумма разницы Treated - Synthetic
    cumulative_effect = np.cumsum(actual - predictions)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(periods, cumulative_effect, label='Кумулятивный эффект', color='purple')

    if treatment_date is not None:
        ax.axvline(x=treatment_date, color='black', linestyle=':', label='Treatment')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    
    if show:
        plt.show()
        
    return fig 

def plot_synthetic_did(model, figsize=(12, 8), title=None, xlabel=None, ylabel=None, grid=True, 
                      confidence_interval=True, alpha=0.05, save_path=None, show=False):
    """
    Визуализация результатов метода Synthetic Difference-in-Differences.
    
    Parameters
    ----------
    model : SyntheticDIDModel
        Обученная модель Synthetic DID
    figsize : tuple, default=(12, 8)
        Размер графика
    title : str, optional
        Заголовок графика
    xlabel : str, optional
        Подпись оси X
    ylabel : str, optional
        Подпись оси Y
    grid : bool, default=True
        Отображать сетку на графике
    confidence_interval : bool, default=True
        Отображать доверительный интервал
    alpha : float, default=0.05
        Уровень значимости для доверительного интервала
    save_path : str, optional
        Путь для сохранения графика
    show : bool, default=False
        Отображать ли график автоматически. В Jupyter Notebook рекомендуется устанавливать False,
        так как график будет отображен автоматически при возврате объекта фигуры.
        
    Returns
    -------
    matplotlib.figure.Figure
        Объект фигуры matplotlib
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not hasattr(model, 'model_'):
        raise ValueError("Модель должна быть обучена перед визуализацией. Вызовите метод fit().")
    
    # Получаем данные из модели
    data = model.data.copy()
    period_col = model.period_index_col
    outcome_col = model.outcome_col
    treat_col = model.treat_col
    post_col = model.post_col
    
    # Вычисляем средние значения для обработанных единиц
    treated_means = (data
                     .query(f"{treat_col}")
                     .groupby([period_col, post_col])[outcome_col]
                     .mean()
                     .reset_index()
                     .sort_values(period_col))
    
    # Вычисляем синтетический контроль
    treated_post = data.query(f"{treat_col} and {post_col}")
    
    # Применяем веса к контрольной группе для построения синтетического контроля
    unit_weights = model.unit_weights_
    time_weights = model.time_weights_
    
    # Получаем контрфактические значения из модели
    joined_data = model.join_weights(data, unit_weights, time_weights)
    counterfactual = joined_data.query(f"{treat_col} and {post_col}").copy()
    counterfactual[treat_col] = 0
    counterfactual['predicted'] = model.model_.predict(counterfactual)
    
    # Группируем контрфактические значения по периодам
    counterfactual_means = (counterfactual
                            .groupby(period_col)['predicted']
                            .mean()
                            .reset_index())
    
    # Создаем график
    fig, ax = plt.subplots(figsize=figsize)
    
    # Определяем период начала воздействия
    treatment_start = data.query(f"{post_col}")[period_col].min()
    
    # Строим линии для фактических и контрфактических значений
    ax.plot(treated_means[period_col], treated_means[outcome_col], 'b-', linewidth=2, label='Фактические значения')
    ax.plot(counterfactual_means[period_col], counterfactual_means['predicted'], 'g--', linewidth=2, label='Синтетический контроль')
    
    # Добавляем вертикальную линию, обозначающую начало воздействия
    ax.axvline(x=treatment_start, color='r', linestyle='--', label='Начало воздействия')
    
    # Вычисляем и отображаем эффект воздействия
    att, se, ci_lower, ci_upper = model.estimate_se(alpha=alpha)
    
    # Заштрихованная область для доверительного интервала
    if confidence_interval and hasattr(model, 'att_'):
        post_periods = sorted(data.query(f"{post_col}")[period_col].unique())
        for period in post_periods:
            # Добавляем точку эффекта
            effect_point = treated_means.query(f"{period_col} == {period}")[outcome_col].values[0]
            counterfactual_point = counterfactual_means.query(f"{period_col} == {period}")['predicted'].values[0]
            effect = effect_point - counterfactual_point
            
            # Рисуем вертикальную линию для эффекта
            ax.plot([period, period], [counterfactual_point, effect_point], 'r-', alpha=0.7)
        
        # Добавляем текст с информацией об эффекте
        ax.text(0.05, 0.95, 
                f"ATT: {att:.4f}\nSE: {se:.4f}\n95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]",
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Настройка графика
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Результаты Synthetic Difference-in-Differences', fontsize=14)
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    else:
        ax.set_xlabel('Период', fontsize=12)
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    else:
        ax.set_ylabel(outcome_col, fontsize=12)
    
    ax.legend(fontsize=12)
    
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Сохраняем график, если указан путь
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Показываем график, если указано
    if show:
        plt.show()
    
    return fig 

def plot_synthetic_diff_in_diff(model, T0, figsize=(14, 7), save_path=None, show=False):
    """
    Визуализация результатов метода Synthetic Difference-in-Differences с подробным графиком.
    
    Parameters
    ----------
    model : SyntheticDIDModel
        Обученная модель Synthetic DID
    T0 : int или float
        Период начала воздействия
    figsize : tuple, default=(14, 7)
        Размер графика
    save_path : str, optional
        Путь для сохранения графика
    show : bool, default=False
        Отображать ли график автоматически. В Jupyter Notebook рекомендуется устанавливать False,
        так как график будет отображен автоматически при возврате объекта фигуры.
        
    Returns
    -------
    matplotlib.figure.Figure
        Объект фигуры matplotlib
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not hasattr(model, 'model_'):
        raise ValueError("Модель должна быть обучена перед визуализацией. Вызовите метод fit().")
    
    try:
        # Получаем необходимые данные из модели
        att = model.att_
        unit_weights = model.unit_weights_
        time_weights = model.time_weights_
        sdid_model_fit = model.model_
        intercept = model.intercept_
        
        # Получаем основные атрибуты из модели
        data = model.data
        outcome_col = model.outcome_col
        period_index_col = model.period_index_col
        shopno_col = model.shopno_col
        treat_col = model.treat_col
        post_col = model.post_col
        
        # Получаем значения для контрольной группы
        y_co_all = data.loc[data[treat_col] == 0] \
                    .pivot_table(index=period_index_col, columns=shopno_col,
                                values=outcome_col, aggfunc='mean') \
                    .sort_index()
        sc_did = intercept + y_co_all.dot(unit_weights)
        
        # Значения для группы воздействия
        treated_all = data.loc[data[treat_col] == 1] \
                        .groupby(period_index_col)[outcome_col].mean()
        
        # Определяем средние периоды до и после воздействия
        pre_times = data.loc[data[period_index_col] < T0, period_index_col]
        post_times = data.loc[data[period_index_col] >= T0, period_index_col]
        avg_pre_period = pre_times.mean() if len(pre_times) > 0 else T0
        avg_post_period = post_times.mean() if len(post_times) > 0 else T0 + 1
        
        # Получаем параметры модели для построения линий тренда
        params = sdid_model_fit.params
        pre_sc = params.get("Intercept", 0)
        post_sc = pre_sc + params.get(post_col, 0)
        pre_treat = pre_sc + params.get(treat_col, 0)
        post_treat = post_sc + params.get(treat_col, 0) + params.get(f"{post_col}:{treat_col}", 0)
        
        # Контрфактическое значение (без эффекта)
        sc_did_y0 = pre_treat + (post_sc - pre_sc)
        
        # Создаем график
        plt.style.use("ggplot")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
        
        # Рисуем все контрольные единицы
        controls_all = data.loc[data[treat_col] == 0]
        for unit in controls_all[shopno_col].unique():
            subset = controls_all.loc[controls_all[shopno_col] == unit].sort_values(period_index_col)
            ax1.plot(subset[period_index_col], subset[outcome_col],
                    color="gray", alpha=0.5, linewidth=1)

        # Рисуем синтетический контроль и группу воздействия
        ax1.plot(sc_did.index, sc_did.values, label="Synthetic DID", color="black", alpha=0.8)
        ax1.plot(treated_all.index, treated_all.values, label="Тестовая группа", color="red", linewidth=2)
        
        # Рисуем линии тренда
        ax1.plot([avg_pre_period, avg_post_period], [pre_sc, post_sc],
                color="C5", label="Синтетический тренд", linewidth=2)
        ax1.plot([avg_pre_period, avg_post_period], [pre_treat, post_treat],
                color="C2", label="Воздействие", linewidth=2)
        ax1.plot([avg_pre_period, avg_post_period], [pre_treat, sc_did_y0],
                color="C2", linestyle="dashed", linewidth=2)
        
        # Добавляем аннотацию для ATT
        x_bracket = avg_post_period
        y_top = post_treat
        y_bottom = sc_did_y0
        ax1.annotate(
            '', 
            xy=(x_bracket, y_bottom), 
            xytext=(x_bracket, y_top),
            arrowprops=dict(arrowstyle='|-|', color='purple', lw=2)
        )
        ax1.text(x_bracket + 0.5, (y_top + y_bottom) / 2, f"ATT = {round(att, 2)}",
                color='purple', fontsize=12, va='center')
        
        # Добавляем легенду и оформление
        ax1.legend()
        ax1.set_title("Синтетический diff-in-diff")
        ax1.axvline(T0, color='black', linestyle=':', label='Начало воздействия')
        ax1.set_ylabel(f"Значение {outcome_col}")

        # Добавляем график весов времени
        ax2.bar(time_weights.index, time_weights.values, color='blue', alpha=0.7)
        ax2.axvline(T0, color="black", linestyle="dotted")
        ax2.set_ylabel("Веса для времени")
        ax2.set_xlabel("Время")
        
        plt.tight_layout()
        
        # Сохраняем график, если указан путь
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Показываем график, если указано
        if show:
            plt.show()
            
        return fig
        
    except Exception as e:
        import traceback
        print(f"Ошибка при построении графика: {str(e)}")
        print(traceback.format_exc())
        return None 