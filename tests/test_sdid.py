import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from synthetic_control import SyntheticDIDModel

class TestSyntheticDIDModel:
    """Tests for SyntheticDIDModel."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data."""
        np.random.seed(42)
        
        n_units = 5
        n_periods = 10
        
        data = []
        for unit in range(n_units):
            for period in range(n_periods):
                is_treated = (unit == 0)
                is_post = period >= 5
                
                metric = 100 + unit * 10 + period * 2 + np.random.normal(0, 5)
                if is_treated and is_post:
                    metric -= 15
                
                data.append({
                    'unit': f"unit_{unit}",
                    'period': f"period_{period}",
                    'metric': metric,
                    'treated': is_treated,
                    'post': is_post
                })
        
        self.df = pd.DataFrame(data)
        
        self.df['unit'] = self.df['unit'].astype('category')
        self.df['period'] = self.df['period'].astype('category')
    
    def test_initialization(self):
        """Test model initialization."""
        model = SyntheticDIDModel(
            data=self.df,
            metric='metric',
            period_index='period',
            unit_id='unit',
            treated='treated',
            after_treatment='post'
        )
        
        assert model.data is not None
        assert model.metric == 'metric'
        assert model.period_index == 'period'
        assert model.unit_id == 'unit'
        assert model.treated == 'treated'
        assert model.after_treatment == 'post'
        assert model.bootstrap_rounds == 100  
        assert model.seed == 42  
    
    def test_fit_and_weights(self):
        """Test model fitting and weights calculation."""
        model = SyntheticDIDModel(
            data=self.df,
            metric='metric',
            period_index='period',
            unit_id='unit',
            treated='treated',
            after_treatment='post',
            bootstrap_rounds=20 
        )
        
        model.fit()
        
        assert model.unit_weights_ is not None
        assert isinstance(model.unit_weights_, pd.Series)
        assert len(model.unit_weights_) > 0
        assert np.isclose(model.unit_weights_.sum(), 1.0)
        
        assert model.time_weights_ is not None
        assert isinstance(model.time_weights_, pd.Series)
        assert len(model.time_weights_) > 0
        assert np.isclose(model.time_weights_.sum(), 1.0)
    
    def test_estimate_effect(self):
        """Test treatment effect estimation."""
        model = SyntheticDIDModel(
            data=self.df,
            metric='metric',
            period_index='period',
            unit_id='unit',
            treated='treated',
            after_treatment='post',
            bootstrap_rounds=20,  
            seed=42
        )
        
        model.fit()
        effect = model.estimate_effect()
        
        assert isinstance(effect, dict)
        assert 'att' in effect
        assert isinstance(effect['att'], (int, float))
        assert 'weights' in effect
        assert isinstance(effect['weights'], pd.Series)
        
        assert -25 < effect['att'] < -5
    
    def test_predict(self):
        """Test counterfactual prediction."""
        model = SyntheticDIDModel(
            data=self.df,
            metric='metric',
            period_index='period',
            unit_id='unit',
            treated='treated',
            after_treatment='post'
        )
        
        model.fit()
        predictions = model.predict()
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(self.df['period'].unique())
        
        treated_data = self.df[(self.df['treated'] == True) & (self.df['post'] == False)]
        pre_treatment_mean = treated_data['metric'].mean()
        
        assert np.all(np.abs(predictions - pre_treatment_mean) < pre_treatment_mean * 0.5)

if __name__ == '__main__':
    pytest.main() 