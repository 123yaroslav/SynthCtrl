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
    ylabel: str = "Metric"
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
    return fig

def plot_effect_distribution(
    effects: np.ndarray,
    observed_effect: float,
    figsize: tuple = (10, 6),
    title: str = "Effect Distribution",
    xlabel: str = "Effect",
    ylabel: str = "Density"
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
    
    return fig

def plot_weights(
    weights: Dict[str, float],
    figsize: tuple = (10, 6),
    title: str = "Control Unit Weights",
    xlabel: str = "Control Unit",
    ylabel: str = "Weight"
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
    ylabel: str = "Кумулятивная разница"
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
    return fig 