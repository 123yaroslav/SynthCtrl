import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from synthetic_control.visualization import (
    plot_synthetic_control,
    plot_effect_distribution,
    plot_weights
)
from .test_base import create_test_data

def test_plot_synthetic_control():
    """Test synthetic control plotting function."""
    data = create_test_data()
    predictions = np.random.normal(100, 10, len(data['date'].unique()))
    
    fig = plot_synthetic_control(
        data=data,
        metric='metric',
        period_index='date',
        unit_id='shop_id',
        treated='treated',
        after_treatment='after_treatment',
        predictions=predictions,
        treatment_date='2020-07-01'
    )
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    
    ax = fig.axes[0]
    assert len(ax.lines) >= 2  
    assert ax.get_title() != ''
    assert ax.get_xlabel() != ''
    assert ax.get_ylabel() != ''
    
    plt.close(fig)

def test_plot_effect_distribution():
    """Test effect distribution plotting function."""
    effects = np.random.normal(0, 1, 1000)
    observed_effect = 0.5
    
    fig = plot_effect_distribution(
        effects=effects,
        observed_effect=observed_effect
    )
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    
    ax = fig.axes[0]
    assert ax.get_title() != ''
    assert ax.get_xlabel() != ''
    assert ax.get_ylabel() != ''
    
    plt.close(fig)

def test_plot_weights():
    """Test weights plotting function."""
    n_weights = 5
    weights = pd.Series(
        np.random.dirichlet(np.ones(n_weights)),
        index=[f'unit_{i}' for i in range(n_weights)]
    )
    
    fig = plot_weights(weights=weights, title='Test Weights')
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    
    ax = fig.axes[0]
    assert len(ax.patches) == n_weights  
    assert ax.get_title() == 'Test Weights'
    assert ax.get_xlabel() != ''
    assert ax.get_ylabel() != ''
    
    plt.close(fig) 