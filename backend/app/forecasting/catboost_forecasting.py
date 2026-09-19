"""
Legacy module for CatBoost forecasting.
Replaced with GRU + Attention forecasting (backend.app.forecasting.gru_forecasting).
"""

from backend.app.forecasting.gru_forecasting import GRUForecasting

# Backwards compatibility alias
CatBoostForecasting = GRUForecasting

__all__ = ["CatBoostForecasting", "GRUForecasting"]
