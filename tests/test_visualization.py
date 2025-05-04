import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from synthetic_control.visualization import (
    plot_synthetic_control,
    plot_effect_distribution,
    plot_weights
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

def test_plot_synthetic_control():
    """Тест функции plot_synthetic_control."""
    data = create_test_data()
    predictions = np.random.normal(100, 10, len(data['date'].unique()))
    
    fig = plot_synthetic_control(
        data=data,
        metric='metric',
        period_index='date',
        shopno='shop_id',
        treated='treated',
        after_treatment='after_treatment',
        predictions=predictions,
        treatment_date='2020-07-01'
    )
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    
    # Проверка наличия всех элементов графика
    ax = fig.axes[0]
    assert len(ax.lines) == 3  # treated, synthetic control, treatment line
    assert ax.get_title() == "Synthetic Control"
    assert ax.get_xlabel() == "Date"
    assert ax.get_ylabel() == "Metric"
    assert ax.get_legend() is not None

def test_plot_effect_distribution():
    """Тест функции plot_effect_distribution."""
    effects = np.random.normal(0, 1, 1000)
    observed_effect = 0.5
    
    fig = plot_effect_distribution(
        effects=effects,
        observed_effect=observed_effect
    )
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    
    # Проверка наличия всех элементов графика
    ax = fig.axes[0]
    assert len(ax.lines) == 1  # observed effect line
    assert ax.get_title() == "Effect Distribution"
    assert ax.get_xlabel() == "Effect"
    assert ax.get_ylabel() == "Density"
    assert ax.get_legend() is not None

def test_plot_weights():
    """Тест функции plot_weights."""
    weights = {
        f'shop_{i}': np.random.random() for i in range(5)
    }
    
    fig = plot_weights(weights)
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    
    # Проверка наличия всех элементов графика
    ax = fig.axes[0]
    assert len(ax.patches) == len(weights)  # bars
    assert ax.get_title() == "Control Unit Weights"
    assert ax.get_xlabel() == "Control Unit"
    assert ax.get_ylabel() == "Weight" 