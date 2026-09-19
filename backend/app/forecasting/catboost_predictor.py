"""
Legacy module for CatBoost predictor.
Replaced with GRUPredictor (backend.app.forecasting.predictor).
"""

from backend.app.forecasting.predictor import GRUPredictor
from backend.app.forecasting.gru_forecasting import GRUForecasting

# Backwards compatibility aliases
CatBoostPredictor = GRUPredictor

__all__ = ["CatBoostPredictor", "GRUPredictor", "GRUForecasting"]
