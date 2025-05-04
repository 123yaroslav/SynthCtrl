import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional
from scipy.optimize import fmin_slsqp
from functools import partial

class SyntheticControl:
    """
    Базовый класс для реализации метода Synthetic Control.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame с данными
    metric : str
        Название метрики для анализа
    period_index : str
        Название колонки с временными периодами
    shopno : str
        Название колонки с идентификаторами магазинов
    treated : str
        Название колонки, указывающей на обработанные единицы
    after_treatment : str
        Название колонки, указывающей на периоды после вмешательства
    bootstrap_rounds : int, default=100
        Количество раундов бутстрепа для оценки стандартной ошибки
    seed : int, default=42
        Seed для воспроизводимости результатов
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        metric: str,
        period_index: str,
        shopno: str,
        treated: str,
        after_treatment: str,
        bootstrap_rounds: int = 100,
        seed: int = 42
    ):
        self.data = data.copy()
        self.metric = metric
        self.period_index = period_index
        self.shopno = shopno
        self.treated = treated
        self.after_treatment = after_treatment
        self.bootstrap_rounds = bootstrap_rounds
        self.seed = seed
        
        # Проверка входных данных
        self._validate_input()
        
    def _validate_input(self) -> None:
        """Проверка корректности входных данных."""
        required_columns = [
            self.metric,
            self.period_index,
            self.shopno,
            self.treated,
            self.after_treatment
        ]
        
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют необходимые колонки: {missing_columns}")
            
    def loss(self, W: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Функция потерь для оптимизации весов.
        
        Parameters
        ----------
        W : np.ndarray
            Вектор весов
        X : np.ndarray
            Матрица признаков контрольных единиц
        y : np.ndarray
            Вектор значений целевой переменной для обработанной единицы
        
        Returns
        -------
        float
            Значение функции потерь
        """
        # Проверка размерностей
        if len(y) == 0 or len(X) == 0:
            return np.inf
        
        # Проверка совпадения размерностей
        if X.shape[0] != len(y):
            raise ValueError(f"Несоответствие размерностей: X.shape[0]={X.shape[0]}, len(y)={len(y)}")
        
        # Проверка количества признаков
        if X.shape[1] != len(W):
            raise ValueError(f"Несоответствие размерностей: X.shape[1]={X.shape[1]}, len(W)={len(W)}")
        
        return np.sqrt(np.mean((y - X.dot(W))**2))
        
    def fit(self) -> None:
        """Обучение модели Synthetic Control."""
        raise NotImplementedError("Метод должен быть реализован в подклассах")
        
    def predict(self) -> np.ndarray:
        """Предсказание значений для обработанных единиц."""
        raise NotImplementedError("Метод должен быть реализован в подклассах")
        
    def estimate_effect(self) -> Dict[str, float]:
        """Оценка эффекта вмешательства."""
        raise NotImplementedError("Метод должен быть реализован в подклассах")
        
    def bootstrap_effect(self) -> Dict[str, float]:
        """Оценка стандартной ошибки эффекта с помощью бутстрепа."""
        raise NotImplementedError("Метод должен быть реализован в подклассах") 