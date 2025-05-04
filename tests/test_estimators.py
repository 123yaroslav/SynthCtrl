import pytest
import numpy as np
import pandas as pd
from synthetic_control.estimators import ClassicSyntheticControl

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

def test_classic_synthetic_control_initialization():
    """Тест инициализации ClassicSyntheticControl."""
    data = create_test_data()
    sc = ClassicSyntheticControl(
        data=data,
        metric='metric',
        period_index='date',
        shopno='shop_id',
        treated='treated',
        after_treatment='after_treatment'
    )
    
    assert sc.weights_ is None
    assert sc.control_units_ is None
    assert sc.X_ is None
    assert sc.y_ is None

def test_fit():
    """Тест метода fit."""
    data = create_test_data()
    sc = ClassicSyntheticControl(
        data=data,
        metric='metric',
        period_index='date',
        shopno='shop_id',
        treated='treated',
        after_treatment='after_treatment'
    )
    
    sc.fit()
    
    assert sc.weights_ is not None
    assert sc.control_units_ is not None
    assert sc.X_ is not None
    assert sc.y_ is not None
    assert len(sc.weights_) == len(sc.control_units_)
    assert np.isclose(np.sum(sc.weights_), 1.0)
    assert np.all(sc.weights_ >= 0)
    assert np.all(sc.weights_ <= 1)

def test_predict():
    """Тест метода predict."""
    data = create_test_data()
    sc = ClassicSyntheticControl(
        data=data,
        metric='metric',
        period_index='date',
        shopno='shop_id',
        treated='treated',
        after_treatment='after_treatment'
    )
    
    # Проверка ошибки при вызове predict до fit
    with pytest.raises(ValueError):
        sc.predict()
    
    sc.fit()
    predictions = sc.predict()
    
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(data[data.period_index].unique())

def test_estimate_effect():
    """Тест метода estimate_effect."""
    data = create_test_data()
    sc = ClassicSyntheticControl(
        data=data,
        metric='metric',
        period_index='date',
        shopno='shop_id',
        treated='treated',
        after_treatment='after_treatment'
    )
    
    # Проверка ошибки при вызове estimate_effect до fit
    with pytest.raises(ValueError):
        sc.estimate_effect()
    
    sc.fit()
    effect = sc.estimate_effect()
    
    assert isinstance(effect, dict)
    assert 'att' in effect
    assert 'weights' in effect
    assert isinstance(effect['att'], float)
    assert isinstance(effect['weights'], dict)

def test_bootstrap_effect():
    """Тест метода bootstrap_effect."""
    data = create_test_data()
    sc = ClassicSyntheticControl(
        data=data,
        metric='metric',
        period_index='date',
        shopno='shop_id',
        treated='treated',
        after_treatment='after_treatment'
    )
    
    # Проверка ошибки при вызове bootstrap_effect до fit
    with pytest.raises(ValueError):
        sc.bootstrap_effect()
    
    sc.fit()
    bootstrap_results = sc.bootstrap_effect()
    
    assert isinstance(bootstrap_results, dict)
    assert 'se' in bootstrap_results
    assert 'ci_lower' in bootstrap_results
    assert 'ci_upper' in bootstrap_results
    assert isinstance(bootstrap_results['se'], float)
    assert isinstance(bootstrap_results['ci_lower'], float)
    assert isinstance(bootstrap_results['ci_upper'], float)
    assert bootstrap_results['ci_lower'] <= bootstrap_results['ci_upper'] 