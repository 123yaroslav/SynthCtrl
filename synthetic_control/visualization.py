import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional
import seaborn as sns

def plot_synthetic_control(
    data: pd.DataFrame,
    metric: str,
    period_index: str,
    unit_id: str,
    treated: str,
    after_treatment: str,
    predictions: np.ndarray,
    treatment_date: Optional[int] = None,
    figsize: tuple = (12, 6),
    title: str = "Synthetic Control",
    xlabel: str = "Date",
    ylabel: str = "Metric",
    show: bool = False
) -> plt.Figure:
    """
    Visualization of Synthetic Control results.
    
    Parameters
    ----------
    data : pd.DataFrame
        Original data
    metric : str
        Metric name
    period_index : str
        Name of the period column
    unit_id : str
        Name of the unit identifier column
    treated : str
        Name of the column indicating treated units
    after_treatment : str
        Name of the column indicating periods after intervention
    predictions : np.ndarray
        Predicted values
    treatment_date : Optional[int]
        Intervention date
    figsize : tuple, default=(12, 6)
        Figure size
    title : str, default="Synthetic Control"
        Figure title
    xlabel : str, default="Date"
        X-axis label
    ylabel : str, default="Metric"
        Y-axis label
    show : bool, default=False
        Whether to show the figure automatically
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    treated_data = data[data[treated]].sort_values(period_index)
    periods = treated_data[period_index].values
    actual = treated_data[metric].values

    if len(predictions) != len(actual):
        if len(predictions) > len(actual):
            predictions = predictions[-len(actual):]
        else:
            raise ValueError("Length of predictions is less than length of actual data for treated unit.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(periods, actual, label='Treated', color='blue')
    ax.plot(periods, predictions, label='Synthetic Control', color='red', linestyle='--')

    if treatment_date is not None:
        ax.axvline(x=treatment_date, color='black', linestyle=':', label='Treatment')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    
    if show:
        plt.show()
        
    return fig

def plot_effect_distribution(
    effects: np.ndarray,
    observed_effect: float,
    figsize: tuple = (10, 6),
    title: str = "Effect Distribution",
    xlabel: str = "Effect",
    ylabel: str = "Density",
    show: bool = False
) -> plt.Figure:
    """
    Visualization of effect distribution.
    
    Parameters
    ----------
    effects : np.ndarray
        Array of effects
    observed_effect : float
        Observed effect
    figsize : tuple, default=(10, 6)
        Figure size
    title : str, default="Effect Distribution"
        Figure title
    xlabel : str, default="Effect"
        X-axis label
    ylabel : str, default="Density"
        Y-axis label
    show : bool, default=False
        Whether to show the figure automatically
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.histplot(effects, kde=True, ax=ax)
    
    ax.axvline(x=observed_effect, color='red', linestyle='--', label='Observed Effect')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    
    if show:
        plt.show()
    
    return fig

def plot_weights(
    weights: Dict[str, float],
    figsize: tuple = (10, 6),
    title: str = "Control Unit Weights",
    xlabel: str = "Control Unit",
    ylabel: str = "Weight",
    show: bool = False
) -> plt.Figure:
    """
    Visualization of control unit weights.
    
    Parameters
    ----------
    weights : Dict[str, float]
        Dictionary with control unit weights
    figsize : tuple, default=(10, 6)
        Figure size
    title : str, default="Control Unit Weights"
        Figure title
    xlabel : str, default="Control Unit"
        X-axis label
    ylabel : str, default="Weight"
        Y-axis label
    show : bool, default=False
        Whether to show the figure automatically
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    units = list(weights.keys())
    values = list(weights.values())
    
    ax.bar(units, values)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    
    plt.xticks(rotation=45)
    
    if show:
        plt.show()
    
    return fig 

def plot_cumulative_effect(
    data: pd.DataFrame,
    metric: str,
    period_index: str,
    treated: str,
    predictions: np.ndarray,
    treatment_date: Optional[int] = None,
    figsize: tuple = (12, 6),
    title: str = "Cumulative Effect of Synthetic Control",
    xlabel: str = "Date",
    ylabel: str = "Cumulative Difference",
    show: bool = False
) -> plt.Figure:
    """
    Visualization of cumulative effect between Treated and Synthetic Control.
    
    Parameters
    ----------
    data : pd.DataFrame
        Original data
    metric : str
        Metric name
    period_index : str
        Name of the period column
    treated : str
        Name of the column indicating treated units
    predictions : np.ndarray
        Predicted values (Synthetic Control)
    treatment_date : Optional[int]
        Treatment date
    figsize : tuple, default=(12, 6)
        Figure size
    title : str, default="Cumulative Effect of Synthetic Control"
        Figure title
    xlabel : str, default="Date"
        X-axis label
    ylabel : str, default="Cumulative Difference"
        Y-axis label
    show : bool, default=False
        Whether to show the figure automatically
    
    Returns
    -------
    plt.Figure
        Figure object
    """
    treated_data = data[data[treated]].sort_values(period_index)
    periods = treated_data[period_index].values
    actual = treated_data[metric].values

    try:
        if len(predictions) != len(actual):
            if len(predictions) > len(actual):
                all_periods = sorted(data[period_index].unique())
                if len(all_periods) == len(predictions):
                    pred_df = pd.DataFrame({
                        'period': all_periods,
                        'predicted': predictions
                    }).set_index('period')
                    
                    predictions = np.array([pred_df.loc[p, 'predicted'] 
                                         if p in pred_df.index else np.nan 
                                         for p in periods])
                else:
                    predictions = predictions[-len(actual):]
            else:
                control_periods = data[~data[treated]].sort_values(period_index)[period_index].unique()
                if len(control_periods) == len(predictions):
                    pred_df = pd.DataFrame({
                        'period': control_periods,
                        'predicted': predictions
                    }).set_index('period')
                    
                    full_predictions = np.full(len(actual), np.nan)
                    for i, period in enumerate(periods):
                        if period in pred_df.index:
                            full_predictions[i] = pred_df.loc[period, 'predicted']
                    
                    if not np.any(~np.isnan(full_predictions)):
                        raise ValueError("No predictions for treated group periods.")
                    
                    predictions = full_predictions
                else:
                    raise ValueError("Length of predictions is less than length of actual data for treated unit.")
        
        if np.any(np.isnan(predictions)):
            valid_indices = ~np.isnan(predictions)
            if not np.any(valid_indices):
                raise ValueError("All predictions contain NaN")
            
            filtered_periods = periods[valid_indices]
            filtered_actual = actual[valid_indices]
            filtered_predictions = predictions[valid_indices]
            
            cumulative_effect = np.cumsum(filtered_actual - filtered_predictions)
            
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(filtered_periods, cumulative_effect, label='Cumulative Effect', color='purple')
        else:
            cumulative_effect = np.cumsum(actual - predictions)
            
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(periods, cumulative_effect, label='Cumulative Effect', color='purple')
        
        if treatment_date is not None:
            ax.axvline(x=treatment_date, color='black', linestyle=':', label='Treatment')
    
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
        
        if show:
            plt.show()
            
        return fig
    
    except Exception as e:
        print(f"Error building cumulative effect plot: {str(e)}")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Error building plot: {str(e)}", 
                ha='center', va='center', transform=ax.transAxes)
        if show:
            plt.show()
        return fig

def plot_model_results(model, T0=None, figsize=(14, 7), save_path=None, show=False):
    """
    Universal function for visualizing results of Synthetic Control and Synthetic DID models.
    
    Parameters
    ----------
    model : SyntheticControl or its subclasses (ClassicSyntheticControl, SyntheticDIDModel)
        Trained model
    T0 : int or float, optional
        Period of intervention start. If None, it's taken from the model.
    figsize : tuple, default=(14, 7)
        Figure size
    save_path : str, optional
        Path to save the figure
    show : bool, default=False
        Whether to show the figure automatically
        
    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from .estimators import ClassicSyntheticControl, SyntheticDIDModel
    
    plt.close('all')
    
    if not hasattr(model, 'data'):
        raise ValueError("Model must be trained before visualization. Call fit() method.")
    
    if isinstance(model, SyntheticDIDModel):
        return _plot_synthetic_diff_in_diff_model(model, T0, figsize, save_path, show)
    else:
        return _plot_classic_synthetic_control(model, T0, figsize, save_path, show)

def _plot_classic_synthetic_control(model, T0=None, figsize=(14, 7), save_path=None, show=False):
    """
    Visualization for Classic Synthetic Control model.
    
    Parameters
    ----------
    model : SyntheticControl or ClassicSyntheticControl
        Trained model
    T0 : int or float, optional
        Treatment period start. If None, it's taken from the model.
    figsize : tuple, default=(14, 7)
        Figure size
    save_path : str, optional
        Path to save the figure
    show : bool, default=False
        Whether to show the figure automatically
        
    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    data = model.data.copy()
    outcome_col = model.metric
    period_index_col = model.period_index
    unit_id_col = model.unit_id
    treat_col = model.treated
    post_col = model.after_treatment
    
    if T0 is None:
        if hasattr(model, 'treatment_date') and model.treatment_date is not None:
            T0 = model.treatment_date
        else:
            T0 = data[data[post_col]].sort_values(period_index_col)[period_index_col].min()
            print(f"Warning: Treatment date not specified, using first post-treatment period: {T0}")
    
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=figsize)
    
    treated_data = data[data[treat_col]].sort_values(period_index_col)
    treated_values = treated_data.groupby(period_index_col)[outcome_col].mean()
    
    ax.plot(treated_values.index, treated_values.values, 
            label="Treatment Group", color="red", linewidth=2)
    
    try:
        predictions = model.predict()
        
        if len(predictions) < len(treated_values):
            control_periods = data[~data[treat_col]].sort_values(period_index_col)[period_index_col].unique()
            control_periods = [p for p in control_periods if p in treated_values.index]
            
            if len(control_periods) == len(predictions):
                pred_df = pd.DataFrame({
                    'period': control_periods,
                    'predicted': predictions
                }).set_index('period')
                
                full_predictions = np.full(len(treated_values), np.nan)
                for i, period in enumerate(treated_values.index):
                    if period in pred_df.index:
                        full_predictions[i] = pred_df.loc[period, 'predicted']
                
                predictions = full_predictions
                if np.isnan(predictions).any():
                    print("Warning: Some periods have no predictions.")
        
        elif len(predictions) > len(treated_values):
            all_periods = sorted(data[period_index_col].unique())
            
            if len(all_periods) == len(predictions):
                pred_df = pd.DataFrame({
                    'period': all_periods,
                    'predicted': predictions
                }).set_index('period')
                
                predictions = np.array([pred_df.loc[p, 'predicted'] if p in pred_df.index else np.nan 
                                       for p in treated_values.index])
            else:
                predictions = predictions[-len(treated_values):]
        
        ax.plot(treated_values.index, predictions, 
                label="Synthetic Control", color="blue", linestyle="--", linewidth=2)
        
        ax.axvline(x=T0, color='black', linestyle=':', label='Treatment Start')
        
        post_periods = [p for p in treated_values.index if p >= T0]
        if post_periods:
            post_treated = treated_values.loc[post_periods]
            post_pred_indices = [i for i, p in enumerate(treated_values.index) if p in post_periods]
            post_predicted = predictions[post_pred_indices]
            
            valid_indices = ~np.isnan(post_predicted)
            if np.any(valid_indices):
                att = np.mean(np.array(post_treated)[valid_indices] - post_predicted[valid_indices])
                
                ax.text(0.05, 0.95, 
                       f"ATT: {att:.4f}",
                       transform=ax.transAxes, fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                for i, period in enumerate(post_periods):
                    if i < len(post_predicted) and not np.isnan(post_predicted[i]):
                        actual = post_treated.iloc[i]
                        pred = post_predicted[i]
                        ax.plot([period, period], [pred, actual], 'r-', alpha=0.7)
    
    except Exception as e:
        print(f"Error building predictions plot: {str(e)}")
    
    ax.set_title("Synthetic Control Results", fontsize=14)
    ax.set_xlabel(period_index_col, fontsize=12)
    ax.set_ylabel(outcome_col, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
        
    return fig

def _plot_synthetic_diff_in_diff_model(model, T0=None, figsize=(14, 7), save_path=None, show=False):
    """
    Visualization for Synthetic DID model.
    
    Parameters
    ----------
    model : SyntheticDIDModel
        Trained Synthetic DID model
    T0 : int or float, optional
        Treatment period start. If None, it's taken from the model.
    figsize : tuple, default=(14, 7)
        Figure size
    save_path : str, optional
        Path to save the figure
    show : bool, default=False
        Whether to show the figure automatically
        
    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not hasattr(model, 'model_'):
        raise ValueError("Model must be trained before visualization. Call fit() method")
    
    if T0 is None:
        if hasattr(model, 'treatment_date') and model.treatment_date is not None:
            T0 = model.treatment_date
        else:
            T0 = model.data[model.data[model.post_col]].sort_values(model.period_index_col)[model.period_index_col].min()
            print(f"Warning: Treatment date not specified, using first post-treatment period: {T0}")
    
    try:
        att, unit_weights, time_weights, sdid_model_fit, intercept = model.synthetic_diff_in_diff()
        
        y_co_all = model.data.loc[model.data[model.treat_col] == 0] \
                      .pivot(index=model.period_index_col, columns=model.shopno_col, values=model.outcome_col) \
                      .sort_index()
        sc_did = intercept + y_co_all.dot(unit_weights)
        
        treated_all = model.data.loc[model.data[model.treat_col] == 1] \
                          .groupby(model.period_index_col)[model.outcome_col].mean()
        
        pre_times = model.data.loc[model.data[model.period_index_col] < T0, model.period_index_col]
        post_times = model.data.loc[model.data[model.period_index_col] >= T0, model.period_index_col]
        avg_pre_period = pre_times.mean() if len(pre_times) > 0 else T0
        avg_post_period = post_times.mean() if len(post_times) > 0 else T0 + 1
        
        params = sdid_model_fit.params
        pre_sc = params.get("Intercept", 0)
        post_sc = pre_sc + params.get(model.post_col, 0)
        pre_treat = pre_sc + params.get(model.treat_col, 0)
        
        post_treat_key = f"{model.post_col}:{model.treat_col}"
        if post_treat_key in params:
            post_treat = post_sc + params[model.treat_col] + params[post_treat_key]
        else:
            post_treat = pre_treat
        
        sc_did_y0 = pre_treat + (post_sc - pre_sc)
        
        plt.style.use("ggplot")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.plot(sc_did.index, sc_did.values, label="Synthetic DID", color="black", alpha=0.8)
        ax1.plot(treated_all.index, treated_all.values, label="Treatment Group", color="red", linewidth=2)
        
        ax1.plot([avg_pre_period, avg_post_period], [pre_sc, post_sc],
                 color="#1f77b4", label="Counterfactual Trend", linewidth=2)
        ax1.plot([avg_pre_period, avg_post_period], [pre_treat, post_treat],
                 color="#ff7f0e", linestyle="dashed", label="Effect", linewidth=2)
        ax1.plot([avg_pre_period, avg_post_period], [pre_treat, sc_did_y0],
                 color="#ff7f0e", label="Synthetic Trend", linewidth=2)
        
        x_bracket = avg_post_period
        y_top = post_treat
        y_bottom = sc_did_y0
        ax1.annotate(
            '', 
            xy=(x_bracket, y_bottom), 
            xytext=(x_bracket, y_top),
            arrowprops=dict(arrowstyle='|-|', color='#9467bd', lw=2)
        )
        
        ax1.text(x_bracket + 0.5, (y_top + y_bottom) / 2, f"ATT = {round(att, 4)}",
                 color='black', fontsize=12, va='center',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.legend()
        ax1.set_title("Synthetic Difference-in-Differences")
        ax1.axvline(T0, color='black', linestyle=':', label='Treatment Start')
        ax1.set_ylabel(model.outcome_col)

        ax2.bar(time_weights.index, time_weights.values, color='blue', alpha=0.7)
        ax2.axvline(T0, color="black", linestyle="dotted")
        ax2.set_ylabel("Time Weights")
        ax2.set_xlabel(model.period_index_col)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
            
        return fig
    
    except Exception as e:
        print(f"Error building Synthetic DID plot: {str(e)}")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Error building plot: {str(e)}", 
               ha='center', va='center', transform=ax.transAxes)
        if show:
            plt.show()
        return fig

def plot_synthetic_diff_in_diff(model, T0=None, figsize=(14, 7), save_path=None, show=False):
    """
    Visualization of Synthetic Difference-in-Differences method results.
    
    Parameters
    ----------
    model : SyntheticDIDModel or ClassicSyntheticControl
        Trained model
    T0 : int or float, optional
        Period of intervention start. If None, it's taken from the model.
    figsize : tuple, default=(14, 7)
        Figure size
    save_path : str, optional
        Path to save the figure
    show : bool, default=False
        Whether to show the figure automatically
        
    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure object
    """
    return plot_model_results(model, T0, figsize, save_path, show) 