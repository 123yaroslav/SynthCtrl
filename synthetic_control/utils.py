import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from scipy.stats import norm

def validate_data(
    data: pd.DataFrame,
    required_columns: List[str]
) -> None:
    """
    Validate the presence of required columns in the data.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with data
    required_columns : List[str]
        List of required columns
        
    Raises
    ------
    ValueError
        If required columns are missing
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate RMSE between true and predicted values.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    float
        RMSE value
    """
    return np.sqrt(np.mean((y_true - y_pred)**2))

def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² between true and predicted values.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    float
        R² value
    """
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def calculate_confidence_intervals(
    effects: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Calculate confidence intervals for effects.
    
    Parameters
    ----------
    effects : np.ndarray
        Array of effects
    alpha : float, default=0.05
        Significance level
        
    Returns
    -------
    Dict[str, float]
        Dictionary with confidence interval bounds
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
    Prepare data for Synthetic Control.
    
    Parameters
    ----------
    data : pd.DataFrame
        Original data
    metric : str
        Metric name
    period_index : str
        Name of the period column
    shopno : str
        Name of the unit identifier column
    treated : str
        Name of the column indicating treated units
    after_treatment : str
        Name of the column indicating periods after intervention
        
    Returns
    -------
    Dict[str, Union[pd.DataFrame, np.ndarray]]
        Dictionary with prepared data
    """
    required_columns = [metric, period_index, shopno, treated, after_treatment]
    validate_data(data, required_columns)
    
    df_pre_control = (data
        .query(f"not {treated}")
        .query(f"not {after_treatment}")
        .pivot(index=period_index,
               columns=shopno,
               values=metric)
    )
    
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