import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from synthetic_control.visualization import (
    plot_synthetic_control,
    plot_effect_distribution,
    plot_weights
)

def test_visualization_functions():
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
    predictions = np.random.normal(100, 10, len(dates))
    
    fig = plot_synthetic_control(
        data=data,
        metric='metric',
        period_index='date',
        unit_id='shop_id',
        treated='treated',
        after_treatment='after_treatment',
        predictions=predictions,
        treatment_date='2020-07-01',
        show=False
    )
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    effects = np.random.normal(0, 1, 100)
    observed_effect = 0.5
    
    fig = plot_effect_distribution(
        effects=effects,
        observed_effect=observed_effect,
        show=False
    )
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    weights = pd.Series(
        np.random.dirichlet(np.ones(3)),
        index=[f'unit_{i}' for i in range(3)]
    )
    
    fig = plot_weights(
        weights=weights,
        show=False
    )
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig) 