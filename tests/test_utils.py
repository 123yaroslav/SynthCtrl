import pytest
import numpy as np
import pandas as pd
from synthetic_control.utils import (
    validate_data,
    calculate_rmse,
    calculate_r2,
    calculate_confidence_intervals
)

def test_utils():
    np.random.seed(42)
    
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
    
    rmse = calculate_rmse(y_true, y_pred)
    assert isinstance(rmse, float)
    assert rmse > 0
    
    r2 = calculate_r2(y_true, y_pred)
    assert isinstance(r2, float)
    assert 0 <= r2 <= 1
    
    effects = np.random.normal(0, 1, 100)
    ci = calculate_confidence_intervals(effects, 0.05)
    assert ci['ci_lower'] < ci['ci_upper']
    assert 'se' in ci
    
    data = pd.DataFrame({
        'unit': ['A', 'A', 'B', 'B'],
        'period': [1, 2, 1, 2],
        'metric': [10, 20, 15, 25],
        'treated': [True, True, False, False],
        'after': [False, True, False, True]
    })
    
    try:
        validate_data(
            data=data,
            required_columns=['unit', 'period', 'metric', 'treated', 'after']
        )
        assert True
    except ValueError:
        assert False 