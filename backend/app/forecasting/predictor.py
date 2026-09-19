from pathlib import Path

import numpy as np
import pandas as pd
import torch

from backend.app.forecasting.gru_model import GRUAttention


class GRUPredictor:
    """
    Production inference wrapper for the trained
    GRU + Attention RailPulse model.

    This reproduces the preprocessing used by
    src/09_train_gru_attention.py.
    """

    SEQ_LEN = 6

    NUMERIC_FEATURES = [
        "journey_station_index",
        "delay",
        "previous_delay",
        "delay_change",
        "day_of_week",
        "day_of_month",
        "month",
        "is_weekend",
    ]

    CATEGORICAL_FEATURES = [
        "train_no",
        "station_code",
        "previous_station",
    ]

    def __init__(self):

        self.root = Path(".")

        self.data_path = (
            self.root / "data/processed/model_data.csv"
        )

        self.model_path = (
            self.root / "models/gru_attention_best.pt"
        )

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Model data not found: {self.data_path}"
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"GRU checkpoint not found: {self.model_path}"
            )

        # --------------------------------------------------
        # Load original training data
        # --------------------------------------------------

        self.df = pd.read_csv(self.data_path)

        self.df["date"] = pd.to_datetime(
            self.df["date"],
            errors="coerce",
        )

        self.df = self.df.sort_values(
            [
                "train_no",
                "date",
                "journey_station_index",
            ]
        ).reset_index(drop=True)

        # --------------------------------------------------
        # Recreate categorical mappings EXACTLY
        # --------------------------------------------------

        self.cat_maps = {}

        for col in self.CATEGORICAL_FEATURES:

            values = (
                self.df[col]
                .fillna("UNKNOWN")
                .astype(str)
                .unique()
            )

            self.cat_maps[col] = {
                value: i + 1
                for i, value in enumerate(values)
            }

        # --------------------------------------------------
        # Recreate temporal split EXACTLY
        # --------------------------------------------------

        journeys = (
            self.df[
                ["train_no", "date"]
            ]
            .drop_duplicates()
            .sort_values("date")
        )

        n = len(journeys)

        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_journeys = set(
            map(
                tuple,
                journeys.iloc[:train_end].values,
            )
        )

        self.train_journeys = train_journeys

        # --------------------------------------------------
        # Recreate normalization EXACTLY
        # --------------------------------------------------

        self.means = {}
        self.stds = {}

        train_mask = self.df.apply(
            lambda row: (
                row["train_no"],
                row["date"],
            ) in train_journeys,
            axis=1,
        )

        for col in self.NUMERIC_FEATURES:

            mean = self.df.loc[
                train_mask,
                col,
            ].mean()

            std = self.df.loc[
                train_mask,
                col,
            ].std()

            if pd.isna(std) or std == 0:
                std = 1.0

            self.means[col] = float(mean)
            self.stds[col] = float(std)

        # --------------------------------------------------
        # Build exact model architecture
        # --------------------------------------------------

        categorical_sizes = [
            len(self.cat_maps[col])
            for col in self.CATEGORICAL_FEATURES
        ]

        self.device = torch.device("cpu")

        self.model = GRUAttention(
            numeric_dim=len(
                self.NUMERIC_FEATURES
            ),
            categorical_sizes=categorical_sizes,
            hidden_dim=128,
            num_layers=2,
            dropout=0.25,
        ).to(self.device)

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(
            checkpoint
        )

        self.model.eval()

        print()
        print("=" * 50)
        print(" RailPulse AI Forecasting Engine")
        print(" PRIMARY MODEL: GRU + Attention")
        print("=" * 50)
        print(
            f" Sequence length: {self.SEQ_LEN}"
        )
        print(
            f" Numeric features: {len(self.NUMERIC_FEATURES)}"
        )
        print(
            f" Categorical features: "
            f"{len(self.CATEGORICAL_FEATURES)}"
        )
        print(
            f" Checkpoint: {self.model_path}"
        )
        print("=" * 50)
        print()

    # ======================================================
    # PREDICTION
    # ======================================================

    def predict(self, history):

        if history is None:
            return 0.0

        if not isinstance(
            history,
            pd.DataFrame,
        ):
            history = pd.DataFrame(history)

        if history.empty:
            return 0.0

        history = history.copy()

        if "date" in history.columns:
            history["date"] = pd.to_datetime(
                history["date"],
                errors="coerce",
            )

        history = history.sort_values(
            "journey_station_index"
        ).reset_index(drop=True)

        # Training uses six historical observations.
        history = history.tail(
            self.SEQ_LEN
        )

        if len(history) < self.SEQ_LEN:
            return 0.0

        numeric_rows = []
        categorical_rows = []

        for _, row in history.iterrows():

            # ------------------------------------------
            # Numeric
            # ------------------------------------------

            numeric = []

            for col in self.NUMERIC_FEATURES:

                value = row.get(
                    col,
                    0.0,
                )

                try:
                    value = float(value)
                except (
                    TypeError,
                    ValueError,
                ):
                    value = 0.0

                if np.isnan(value):
                    value = self.means.get(
                        col,
                        0.0,
                    )

                # Exact training normalization
                value = (
                    value
                    - self.means[col]
                ) / self.stds[col]

                numeric.append(value)

            numeric_rows.append(numeric)

            # ------------------------------------------
            # Categorical
            # ------------------------------------------

            categorical = []

            for col in self.CATEGORICAL_FEATURES:

                value = row.get(
                    col,
                    "UNKNOWN",
                )

                if pd.isna(value):
                    value = "UNKNOWN"

                value = str(value)

                encoded = self.cat_maps[
                    col
                ].get(
                    value,
                    0,
                )

                categorical.append(
                    encoded
                )

            categorical_rows.append(
                categorical
            )

        # --------------------------------------------------
        # Tensor shapes:
        #
        # numeric     = [1, 6, 8]
        # categorical = [1, 6, 3]
        # --------------------------------------------------

        numeric_tensor = torch.tensor(
            np.array(
                numeric_rows,
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        categorical_tensor = torch.tensor(
            np.array(
                categorical_rows,
                dtype=np.int64,
            ),
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        # --------------------------------------------------
        # Inference
        # --------------------------------------------------

        with torch.no_grad():

            prediction = self.model(
                numeric_tensor,
                categorical_tensor,
            )

        return float(
            prediction.cpu().item()
        )

    # ======================================================
    # INFORMATION
    # ======================================================

    def available_trains(self):

        return sorted(
            self.df["train_no"]
            .astype(str)
            .unique()
            .tolist()
        )

    def has_model(self, train_no):

        return str(train_no) in set(
            self.available_trains()
        )
