import pytest
import numpy as np
import pandas as pd
from synthetic_control.utils import (
    validate_data,
    calculate_rmse,
    calculate_r2,
    calculate_confidence_intervals,
    prepare_data_for_synthetic_control
)

def create_test_data():
    """Создание тестовых данных."""
    np.random.seed(42)
    n_periods = 20
    n_shops = 10
    
    dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='M')
    shop_ids = [f'shop_{i}' for i in range(n_shops)]
    
    data = []
    for date in dates:
        for shop_id in shop_ids:
            data.append({
                'date': date,
                'shop_id': shop_id,
                'metric': np.random.normal(100, 10),
                'treated': shop_id == 'shop_0',
                'after_treatment': date >= pd.Timestamp('2020-07-01')
            })
    
    return pd.DataFrame(data)

def test_validate_data():
    """Тест функции validate_data."""
    data = create_test_data()
    
    # Тест с корректными данными
    validate_data(data, ['date', 'shop_id', 'metric'])
    
    # Тест с отсутствующими колонками
    with pytest.raises(ValueError):
        validate_data(data, ['date', 'shop_id', 'metric', 'missing_column'])

def test_calculate_rmse():
    """Тест функции calculate_rmse."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    
    rmse = calculate_rmse(y_true, y_pred)
    
    assert isinstance(rmse, float)
    assert rmse > 0
    assert np.isclose(rmse, 0.1)

def test_calculate_r2():
    """Тест функции calculate_r2."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    
    r2 = calculate_r2(y_true, y_pred)
    
    assert isinstance(r2, float)
    assert r2 > 0
    assert r2 < 1

def test_calculate_confidence_intervals():
    """Тест функции calculate_confidence_intervals."""
    effects = np.random.normal(0, 1, 1000)
    
    ci = calculate_confidence_intervals(effects)
    
    assert isinstance(ci, dict)
    assert 'se' in ci
    assert 'ci_lower' in ci
    assert 'ci_upper' in ci
    assert isinstance(ci['se'], float)
    assert isinstance(ci['ci_lower'], float)
    assert isinstance(ci['ci_upper'], float)
    assert ci['ci_lower'] <= ci['ci_upper']

def test_prepare_data_for_synthetic_control():
    """Тест функции prepare_data_for_synthetic_control."""
    data = create_test_data()
    
    prepared_data = prepare_data_for_synthetic_control(
        data=data,
        metric='metric',
        period_index='date',
        shopno='shop_id',
        treated='treated',
        after_treatment='after_treatment'
    )
    
    assert isinstance(prepared_data, dict)
    assert 'X' in prepared_data
    assert 'y' in prepared_data
    assert 'control_units' in prepared_data
    assert 'periods' in prepared_data
    assert isinstance(prepared_data['X'], np.ndarray)
    assert isinstance(prepared_data['y'], np.ndarray)
    assert isinstance(prepared_data['control_units'], list)
    assert isinstance(prepared_data['periods'], pd.DatetimeIndex)
    
    # Проверка размерностей
    assert prepared_data['X'].shape[0] == len(prepared_data['y'])
    assert prepared_data['X'].shape[1] == len(prepared_data['control_units'])
    assert len(prepared_data['periods']) == prepared_data['X'].shape[0] 