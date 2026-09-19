from pathlib import Path
from typing import Union, Dict, Any

import numpy as np
import pandas as pd

from backend.app.forecasting.predictor import GRUPredictor


class GRUForecasting:
    """
    Production forecasting service for dynamic railway delay prediction
    using the trained GRU + Attention PyTorch neural network.
    """

    NUMERIC_COLUMNS = GRUPredictor.NUMERIC_FEATURES
    CATEGORICAL_COLUMNS = GRUPredictor.CATEGORICAL_FEATURES
    FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.predictor = GRUPredictor()

    def available_trains(self) -> list:
        return self.predictor.available_trains()

    def has_model(self, train_no: Union[str, int]) -> bool:
        return self.predictor.has_model(train_no)

    def predict(
        self,
        train_no: Union[str, int],
        features_or_history: Union[Dict[str, Any], pd.DataFrame],
    ) -> float:
        """
        Predict downstream delay change for a given train using sequence features.
        Accepts either a single observation dict, single-row DataFrame, or history DataFrame.
        """
        train_no = str(train_no)

        if not self.has_model(train_no):
            raise ValueError(
                f"No GRU model available for train {train_no}. "
                f"Available trains: {self.available_trains()}"
            )

        if isinstance(features_or_history, dict):
            df = pd.DataFrame([features_or_history])
        elif isinstance(features_or_history, pd.DataFrame):
            df = features_or_history.copy()
        else:
            raise TypeError("features_or_history must be a dictionary or pandas DataFrame")

        if "train_no" not in df.columns:
            df["train_no"] = train_no

        # If fewer rows than SEQ_LEN are provided, attempt to supplement from known journey history
        if len(df) < self.predictor.SEQ_LEN:
            if "journey_station_index" in df.columns:
                target_idx = df["journey_station_index"].iloc[-1]
                history_subset = self.predictor.df[
                    (self.predictor.df["train_no"].astype(str) == train_no)
                    & (self.predictor.df["journey_station_index"] <= target_idx)
                ].tail(self.predictor.SEQ_LEN)

                if len(history_subset) >= self.predictor.SEQ_LEN:
                    df = history_subset
                else:
                    # Pad prefix with duplicates of the earliest observation to satisfy SEQ_LEN
                    pad_count = self.predictor.SEQ_LEN - len(df)
                    padding = pd.concat([df.iloc[[0]]] * pad_count, ignore_index=True)
                    df = pd.concat([padding, df], ignore_index=True)
            else:
                pad_count = self.predictor.SEQ_LEN - len(df)
                padding = pd.concat([df.iloc[[0]]] * pad_count, ignore_index=True)
                df = pd.concat([padding, df], ignore_index=True)

        prediction = self.predictor.predict(df)

        if not np.isfinite(prediction):
            return 0.0

        return float(prediction)
