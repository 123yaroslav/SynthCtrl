import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from synthetic_control import SyntheticDIDModel

class TestSyntheticDIDModel(unittest.TestCase):
    
    def setUp(self):
        """Создает тестовые данные перед каждым тестом."""
        # Параметры для генерации данных
        n_units = 10
        n_treated = 2
        n_periods = 8
        treatment_period = 4
        treatment_effect = 2.0
        
        # Генерируем тестовые данные
        np.random.seed(42)
        unit_effects = np.random.normal(0, 1, n_units)
        time_effects = np.arange(n_periods) * 0.5 + np.random.normal(0, 0.3, n_periods)
        
        data = []
        for unit in range(n_units):
            is_treated = unit < n_treated
            for period in range(n_periods):
                is_post = period >= treatment_period
                # Базовое значение метрики
                metric_value = 10 + unit_effects[unit] + time_effects[period] + np.random.normal(0, 0.5)
                
                # Добавляем эффект воздействия
                if is_treated and is_post:
                    metric_value += treatment_effect
                    
                data.append({
                    'shopno': f'unit_{unit}',
                    'period': period,
                    'metric': metric_value,
                    'treated': is_treated,
                    'post': is_post
                })
        
        self.df = pd.DataFrame(data)
        self.true_effect = treatment_effect
    
    def test_initialization(self):
        """Тест инициализации модели."""
        model = SyntheticDIDModel(
            data=self.df,
            metric='metric',
            period_index='period',
            shopno='shopno',
            treated='treated',
            after_treatment='post'
        )
        
        self.assertEqual(model.outcome_col, 'metric')
        self.assertEqual(model.period_index_col, 'period')
        self.assertEqual(model.shopno_col, 'shopno')
        self.assertEqual(model.treat_col, 'treated')
        self.assertEqual(model.post_col, 'post')
        self.assertEqual(model.seed, 42)
        self.assertEqual(model.bootstrap_rounds, 100)
    
    def test_fit_and_weights(self):
        """Тест обучения модели и проверка корректности весов."""
        model = SyntheticDIDModel(
            data=self.df,
            metric='metric',
            period_index='period',
            shopno='shopno',
            treated='treated',
            after_treatment='post',
            bootstrap_rounds=20  # Уменьшаем для скорости теста
        )
        
        model.fit()
        
        # Проверяем, что атрибуты модели корректно созданы после обучения
        self.assertTrue(hasattr(model, 'unit_weights_'))
        self.assertTrue(hasattr(model, 'time_weights_'))
        self.assertTrue(hasattr(model, 'att_'))
        self.assertTrue(hasattr(model, 'model_'))
        
        # Проверяем, что веса единиц и времени корректно нормализованы
        self.assertAlmostEqual(model.unit_weights_.sum(), 1.0, places=6)
        self.assertAlmostEqual(model.time_weights_.sum(), 1.0, places=6)
        
        # Проверяем, что веса неотрицательны
        self.assertTrue((model.unit_weights_ >= 0).all())
        self.assertTrue((model.time_weights_ >= 0).all())
    
    def test_estimate_effect(self):
        """Тест оценки эффекта воздействия."""
        model = SyntheticDIDModel(
            data=self.df,
            metric='metric',
            period_index='period',
            shopno='shopno',
            treated='treated',
            after_treatment='post',
            bootstrap_rounds=20,  # Уменьшаем для скорости теста
            seed=42
        )
        
        effects = model.estimate_effect()
        
        # Проверяем, что оценка эффекта существует
        self.assertIn('att', effects)
        self.assertIn('se', effects)
        self.assertIn('ci_lower', effects)
        self.assertIn('ci_upper', effects)
        self.assertIn('p_value', effects)
        
        # Проверяем, что оценка близка к истинному эффекту (с точностью до 1)
        self.assertLess(abs(effects['att'] - self.true_effect), 1.0)
        
        # Проверяем, что доверительный интервал корректно вычислен
        self.assertLess(effects['ci_lower'], effects['att'])
        self.assertGreater(effects['ci_upper'], effects['att'])
    
    def test_predict(self):
        """Тест предсказания контрфактических значений."""
        model = SyntheticDIDModel(
            data=self.df,
            metric='metric',
            period_index='period',
            shopno='shopno',
            treated='treated',
            after_treatment='post'
        )
        
        model.fit()
        
        # Получаем предсказание
        predictions = model.predict()
        
        # Проверяем, что предсказание возвращает непустой массив
        self.assertGreater(len(predictions), 0)
        
        # Проверяем, что количество предсказаний соответствует количеству обработанных единиц после воздействия
        treated_post_count = len(self.df.query('treated and post'))
        self.assertEqual(len(predictions), treated_post_count)

if __name__ == '__main__':
    unittest.main() 