import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional
from scipy.optimize import fmin_slsqp
from functools import partial

class SyntheticControl:
    """
    Base class for implementing Synthetic Control Method.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with data
    metric : str
        Name of the metric for analysis
    period_index : str
        Name of the column with time periods
    unit_id : str
        Name of the column with unit identifiers
    treated : str
        Name of the column indicating treated units
    after_treatment : str
        Name of the column indicating periods after intervention
    bootstrap_rounds : int, default=100
        Number of bootstrap rounds for standard error estimation
    seed : int, default=42
        Seed for result reproducibility
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        metric: str,
        period_index: str,
        unit_id: str,
        treated: str,
        after_treatment: str,
        bootstrap_rounds: int = 100,
        seed: int = 42
    ):
        self.data = data.copy()
        self.metric = metric
        self.period_index = period_index
        self.unit_id = unit_id
        self.treated = treated
        self.after_treatment = after_treatment
        self.bootstrap_rounds = bootstrap_rounds
        self.seed = seed
        
        self._determine_treatment_date()
        
        self._validate_input()
        
    def _determine_treatment_date(self) -> None:
        """Determine intervention date from the data."""
        try:
            self.treatment_date = self.data[self.data[self.after_treatment]].sort_values(self.period_index)[self.period_index].min()
            if pd.isna(self.treatment_date):
                self.treatment_date = None
                print("Warning: Could not determine treatment date from data.")
        except Exception as e:
            self.treatment_date = None
            print(f"Error determining treatment date: {str(e)}")
        
    def _validate_input(self) -> None:
        """Validate input data correctness."""
        required_columns = [
            self.metric,
            self.period_index,
            self.unit_id,
            self.treated,
            self.after_treatment
        ]
        
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        if self.treatment_date is not None:
            if self.treatment_date not in self.data[self.period_index].unique():
                print(f"Warning: Treatment date {self.treatment_date} not found in data periods.")
                print(f"Available periods: {sorted(self.data[self.period_index].unique())}")
            
    def loss(self, W: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Loss function for weights optimization.
        
        Parameters
        ----------
        W : np.ndarray
            Weight vector
        X : np.ndarray
            Feature matrix of control units
        y : np.ndarray
            Target variable vector for treated unit
        
        Returns
        -------
        float
            Loss function value
        """
        if len(y) == 0 or len(X) == 0:
            return np.inf
        
        if X.shape[0] != len(y):
            raise ValueError(f"Dimension mismatch: X.shape[0]={X.shape[0]}, len(y)={len(y)}")
        
        if X.shape[1] != len(W):
            raise ValueError(f"Dimension mismatch: X.shape[1]={X.shape[1]}, len(W)={len(W)}")
        
        return np.sqrt(np.mean((y - X.dot(W))**2))
        
    def fit(self) -> None:
        """Fit Synthetic Control model."""
        raise NotImplementedError("Method must be implemented in subclasses")
        
    def predict(self) -> np.ndarray:
        """Predict values for treated units."""
        raise NotImplementedError("Method must be implemented in subclasses")
        
    def estimate_effect(self) -> Dict[str, float]:
        """Estimate intervention effect."""
        raise NotImplementedError("Method must be implemented in subclasses")
        
    def bootstrap_effect(self) -> Dict[str, float]:
        """Estimate standard error of the effect using bootstrap."""
        raise NotImplementedError("Method must be implemented in subclasses") 