import pytest
import numpy as np
import pandas as pd
from synthetic_control.base import SyntheticControl

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

def test_initialization():
    """Тест инициализации класса."""
    data = create_test_data()
    sc = SyntheticControl(
        data=data,
        metric='metric',
        period_index='date',
        shopno='shop_id',
        treated='treated',
        after_treatment='after_treatment'
    )
    
    assert sc.metric == 'metric'
    assert sc.period_index == 'date'
    assert sc.shopno == 'shop_id'
    assert sc.treated == 'treated'
    assert sc.after_treatment == 'after_treatment'
    assert sc.bootstrap_rounds == 100
    assert sc.seed == 42

def test_validation():
    """Тест валидации входных данных."""
    data = create_test_data()
    
    # Тест с корректными данными
    sc = SyntheticControl(
        data=data,
        metric='metric',
        period_index='date',
        shopno='shop_id',
        treated='treated',
        after_treatment='after_treatment'
    )
    
    # Тест с отсутствующими колонками
    with pytest.raises(ValueError):
        SyntheticControl(
            data=data.drop(columns=['metric']),
            metric='metric',
            period_index='date',
            shopno='shop_id',
            treated='treated',
            after_treatment='after_treatment'
        )

def test_loss_function():
    """Тест функции потерь."""
    data = create_test_data()
    sc = SyntheticControl(
        data=data,
        metric='metric',
        period_index='date',
        shopno='shop_id',
        treated='treated',
        after_treatment='after_treatment'
    )
    
    # Создаем тестовые данные
    W = np.array([0.5, 0.5])
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1.5, 3.5])
    
    # Проверяем значение функции потерь
    loss = sc.loss(W, X, y)
    assert isinstance(loss, float)
    assert loss >= 0

def test_not_implemented_methods():
    """Тест нереализованных методов."""
    data = create_test_data()
    sc = SyntheticControl(
        data=data,
        metric='metric',
        period_index='date',
        shopno='shop_id',
        treated='treated',
        after_treatment='after_treatment'
    )
    
    # Проверяем, что методы выбрасывают NotImplementedError
    with pytest.raises(NotImplementedError):
        sc.fit()
    
    with pytest.raises(NotImplementedError):
        sc.predict()
    
    with pytest.raises(NotImplementedError):
        sc.estimate_effect()
    
    with pytest.raises(NotImplementedError):
        sc.bootstrap_effect() 