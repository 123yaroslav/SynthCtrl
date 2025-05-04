import pytest
import numpy as np
import pandas as pd
from synthetic_control import ClassicSyntheticControl
from tests.test_base import create_test_data

def test_classic_synthetic_control_initialization():
    """Test ClassicSyntheticControl initialization."""
    data = create_test_data()
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
    assert sc.period_index == 'date'
    assert sc.unit_id == 'shop_id'
    assert sc.treated == 'treated'
    assert sc.after_treatment == 'after_treatment'
    assert sc.weights_ is None
    assert sc.control_units_ is None

def test_fit():
    """Test fit method."""
    data = create_test_data()
    sc = ClassicSyntheticControl(
        data=data,
        metric='metric',
        period_index='date',
        unit_id='shop_id',
        treated='treated',
        after_treatment='after_treatment'
    )
    
    sc.fit()
    
    assert sc.weights_ is not None
    assert isinstance(sc.weights_, pd.Series)
    assert len(sc.weights_) > 0
    assert sc.control_units_ is not None
    assert len(sc.control_units_) > 0

def test_predict():
    """Test predict method."""
    data = create_test_data()
    sc = ClassicSyntheticControl(
        data=data,
        metric='metric',
        period_index='date',
        unit_id='shop_id',
        treated='treated',
        after_treatment='after_treatment'
    )
    
    sc.fit()
    predictions = sc.predict()
    
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) > 0
    
    mean_metric = data['metric'].mean()
    assert np.all(np.abs(predictions - mean_metric) < mean_metric * 2)

def test_estimate_effect():
    """Test estimate_effect method."""
    data = create_test_data()
    sc = ClassicSyntheticControl(
        data=data,
        metric='metric',
        period_index='date',
        unit_id='shop_id',
        treated='treated',
        after_treatment='after_treatment'
    )
    
    sc.fit()
    effect = sc.estimate_effect()
    
    assert isinstance(effect, dict)
    assert 'att' in effect
    assert isinstance(effect['att'], (int, float))
    assert 'weights' in effect
    assert isinstance(effect['weights'], pd.Series)

def test_bootstrap_effect():
    """Test bootstrap_effect method."""
    data = create_test_data()
    sc = ClassicSyntheticControl(
        data=data,
        metric='metric',
        period_index='date',
        unit_id='shop_id',
        treated='treated',
        after_treatment='after_treatment',
        bootstrap_rounds=10  
    )
    
    sc.fit()
    bootstrap_results = sc.bootstrap_effect()
    
    assert isinstance(bootstrap_results, dict)
    assert 'se' in bootstrap_results
    assert isinstance(bootstrap_results['se'], float)
    assert 'ci_lower' in bootstrap_results
    assert isinstance(bootstrap_results['ci_lower'], float)
    assert 'ci_upper' in bootstrap_results
    assert isinstance(bootstrap_results['ci_upper'], float)
    assert bootstrap_results['ci_lower'] <= bootstrap_results['ci_upper'] 