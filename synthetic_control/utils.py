import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from scipy.stats import norm

def validate_data(
    data: pd.DataFrame,
    required_columns: List[str]
) -> None:
    """
    Проверка наличия необходимых колонок в данных.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame с данными
    required_columns : List[str]
        Список необходимых колонок
        
    Raises
    ------
    ValueError
        Если отсутствуют необходимые колонки
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Отсутствуют необходимые колонки: {missing_columns}")

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисление RMSE между истинными и предсказанными значениями.
    
    Parameters
    ----------
    y_true : np.ndarray
        Истинные значения
    y_pred : np.ndarray
        Предсказанные значения
        
    Returns
    -------
    float
        Значение RMSE
    """
    return np.sqrt(np.mean((y_true - y_pred)**2))

def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисление R² между истинными и предсказанными значениями.
    
    Parameters
    ----------
    y_true : np.ndarray
        Истинные значения
    y_pred : np.ndarray
        Предсказанные значения
        
    Returns
    -------
    float
        Значение R²
    """
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def calculate_confidence_intervals(
    effects: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Вычисление доверительных интервалов для эффектов.
    
    Parameters
    ----------
    effects : np.ndarray
        Массив эффектов
    alpha : float, default=0.05
        Уровень значимости
        
    Returns
    -------
    Dict[str, float]
        Словарь с границами доверительного интервала
    """
    se = np.std(effects, ddof=1)
    z = norm.ppf(1 - alpha / 2)
    
    return {
        'se': se,
        'ci_lower': np.mean(effects) - z * se,
        'ci_upper': np.mean(effects) + z * se
    }

def prepare_data_for_synthetic_control(
    data: pd.DataFrame,
    metric: str,
    period_index: str,
    shopno: str,
    treated: str,
    after_treatment: str
) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
    """
    Подготовка данных для Synthetic Control.
    
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
        
    Returns
    -------
    Dict[str, Union[pd.DataFrame, np.ndarray]]
        Словарь с подготовленными данными
    """
    # Проверка наличия необходимых колонок
    required_columns = [metric, period_index, shopno, treated, after_treatment]
    validate_data(data, required_columns)
    
    # Подготовка данных для контрольных единиц
    df_pre_control = (data
        .query(f"not {treated}")
        .query(f"not {after_treatment}")
        .pivot(index=period_index,
               columns=shopno,
               values=metric)
    )
    
    # Подготовка данных для обработанных единиц
    y = (data
        .query(f"not {after_treatment}")
        .query(f"{treated}")
        .groupby(period_index)[metric]
        .mean()
        .values
    )
    
    return {
        'X': df_pre_control.values,
        'y': y,
        'control_units': list(df_pre_control.columns),
        'periods': df_pre_control.index
    }