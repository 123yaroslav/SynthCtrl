import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from synthetic_control import SyntheticControl

def test_initialization():
    np.random.seed(42)
    n_shops = 3
    n_dates = 5
    
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
    
    data = pd.DataFrame(data)
    
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

def test_not_implemented_methods():
    np.random.seed(42)
    n_shops = 3
    n_dates = 5
    
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
    
    data = pd.DataFrame(data)
    
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