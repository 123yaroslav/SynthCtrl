import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from synthetic_control import ClassicSyntheticControl

def test_classic_synthetic_control():
    np.random.seed(42)
    n_shops = 3
    n_dates = 10
    
    dates = pd.date_range(start='2020-01-01', periods=n_dates, freq='M')
    
    data = []
    for i in range(n_shops):
        for date in dates:
            is_treated = (i == 0)  
            is_after = date >= pd.Timestamp('2020-07-01')  
            
            metric_value = np.random.normal(100, 10) - (is_treated and is_after) * 20
            
            data.append({
                'date': date,
                'shop_id': f"shop_{i}",
                'metric': metric_value,
                'treated': is_treated,
                'after_treatment': is_after
            })
    
    data = pd.DataFrame(data)
    
    assert sum((data['treated'] == True) & (data['after_treatment'] == True)) > 0
    
    sc = ClassicSyntheticControl(
        data=data,
        metric='metric',
        period_index='date',
        unit_id='shop_id',
        treated='treated',
        after_treatment='after_treatment'
    )
    
    assert sc.data is not None
    assert sc.metric == 'metric'
    assert sc.unit_id == 'shop_id'
    assert sc.weights_ is None
    
    sc.fit()
    
    assert sc.weights_ is not None
    assert isinstance(sc.weights_, pd.Series)
    
    predictions = sc.predict()
    assert isinstance(predictions, np.ndarray)
    
    effect = sc.estimate_effect()
    assert isinstance(effect, dict)
    assert 'att' in effect
    
    bootstrap_results = sc.bootstrap_effect()
    assert isinstance(bootstrap_results, dict)
    assert 'se' in bootstrap_results 