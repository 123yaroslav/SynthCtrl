import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from synthetic_control import SyntheticControl

def create_test_data():
    """Create test data for SyntheticControl tests."""
    np.random.seed(42)
    n_shops = 10
    n_dates = 20
    
    dates = [datetime(2020, 1, 1) + pd.Timedelta(days=i*15) for i in range(n_dates)]
    
    data = []
    for i in range(n_shops):
        for date in dates:
            is_treated = (i == 0)  
            is_after = date >= datetime(2020, 7, 1)  
            
            metric_value = np.random.normal(100, 10) - (is_treated and is_after) * 20
            
            data.append({
                'date': date,
                'shop_id': f"shop_{i}",
                'metric': metric_value,
                'treated': is_treated,
                'after_treatment': is_after
            })
    
    return pd.DataFrame(data)

def test_initialization():
    """Test SyntheticControl initialization."""
    data = create_test_data()
    sc = SyntheticControl(
        data=data,
        metric='metric',
        period_index='date',
        unit_id='shop_id',
        treated='treated',
        after_treatment='after_treatment'
    )
    
    assert sc.data is not None
    assert sc.metric == 'metric'
    assert sc.period_index == 'date'
    assert sc.unit_id == 'shop_id'
    assert sc.treated == 'treated'
    assert sc.after_treatment == 'after_treatment'
    assert sc.bootstrap_rounds == 100  
    assert sc.seed == 42  

def test_validation():
    """Test input data validation."""
    data = create_test_data()
    
    sc = SyntheticControl(
        data=data,
        metric='metric',
        period_index='date',
        unit_id='shop_id',
        treated='treated',
        after_treatment='after_treatment'
    )
    
    assert sc.data is not None
    
    # Test missing column
    bad_data = data.drop(columns=['metric'])
    with pytest.raises(Exception):
        sc = SyntheticControl(
            data=bad_data,
            metric='metric',
            period_index='date',
            unit_id='shop_id',
            treated='treated',
            after_treatment='after_treatment'
        )

def test_loss_function():
    """Test loss function."""
    data = create_test_data()
    sc = SyntheticControl(
        data=data,
        metric='metric',
        period_index='date',
        unit_id='shop_id',
        treated='treated',
        after_treatment='after_treatment'
    )
    
    W = np.array([0.5, 0.5])
    X = np.array([
        [100, 110],
        [105, 115],
        [110, 120]
    ])
    y = np.array([105, 110, 115])
    
    loss = sc.loss(W, X, y)
    assert isinstance(loss, float)
    assert loss >= 0

def test_not_implemented_methods():
    """Test not implemented methods."""
    data = create_test_data()
    sc = SyntheticControl(
        data=data,
        metric='metric',
        period_index='date',
        unit_id='shop_id',
        treated='treated',
        after_treatment='after_treatment'
    )
    
    with pytest.raises(NotImplementedError):
        sc.fit()
    
    with pytest.raises(NotImplementedError):
        sc.predict()
    
    with pytest.raises(NotImplementedError):
        sc.estimate_effect()
    
    with pytest.raises(NotImplementedError):
        sc.bootstrap_effect() 