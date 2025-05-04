import pytest
import numpy as np
import pandas as pd
from synthetic_control import SyntheticDIDModel

def test_synthetic_did_model():
    np.random.seed(42)
    
    n_units = 3
    n_periods = 6
    
    data = []
    for unit in range(n_units):
        for period in range(n_periods):
            is_treated = (unit == 0)
            is_post = period >= 3
            
            metric = 100 + unit * 10 + period * 2 + np.random.normal(0, 5)
            if is_treated and is_post:
                metric -= 15
            
            data.append({
                'unit': f"unit_{unit}",
                'period': period,
                'metric': metric,
                'treated': is_treated,
                'post': is_post
            })
    
    df = pd.DataFrame(data)
    
    assert sum((df['treated'] == True) & (df['post'] == False)) > 0
    assert sum((df['treated'] == True) & (df['post'] == True)) > 0
    
    assert sum(df['treated'] == False) > 0
    
    model = SyntheticDIDModel(
        data=df,
        metric='metric',
        period_index='period',
        unit_id='unit',
        treated='treated',
        after_treatment='post',
        bootstrap_rounds=2
    )
    
    assert model.data is not None
    assert model.metric == 'metric'
    assert model.unit_id == 'unit'
    
    model.fit()
    
    assert model.unit_weights_ is not None
    assert isinstance(model.unit_weights_, pd.Series)
    
    assert model.time_weights_ is not None
    assert isinstance(model.time_weights_, pd.Series)
    
    effect = model.estimate_effect()
    assert isinstance(effect, dict)
    assert 'att' in effect
    
    predictions = model.predict()
    assert isinstance(predictions, np.ndarray)

if __name__ == '__main__':
    pytest.main() 