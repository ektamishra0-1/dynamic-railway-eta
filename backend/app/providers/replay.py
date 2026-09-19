from pathlib import Path

import pandas as pd

from backend.app.providers.base import RailwayDataProvider
from backend.app.schemas import TrainObservation


ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "model_data.csv"
)


class ReplayProvider(RailwayDataProvider):

    def __init__(self):

        self.df = pd.read_csv(
            DATA_PATH
        )

        self.df["date"] = pd.to_datetime(
            self.df["date"]
        ).dt.strftime("%Y-%m-%d")

        self.df = self.df.sort_values(
            [
                "train_no",
                "date",
                "journey_station_index",
            ]
        )


    def get_current_observation(
        self,
        train_no: str,
        journey_date: str,
    ):

        journey = self.df[
            (self.df["train_no"].astype(str) == str(train_no))
            &
            (self.df["date"] == journey_date)
        ]

        if journey.empty:
            raise ValueError(
                f"Journey not found: "
                f"{train_no} / {journey_date}"
            )

        row = journey.iloc[
            min(5, len(journey) - 1)
        ]

        return TrainObservation(

            train_no=str(
                row["train_no"]
            ),

            journey_date=str(
                row["date"]
            ),

            station_code=str(
                row["station_code"]
            ),

            station_name=str(
                row.get(
                    "station_name",
                    "",
                )
            ),

            station_sequence=int(
                row["journey_station_index"]
            ),

            delay=float(
                row["delay"]
            ),

            previous_delay=float(
                row.get(
                    "previous_delay",
                    0,
                )
            ),

            delay_change=float(
                row.get(
                    "delay_change",
                    0,
                )
            ),
        )


    def get_downstream_stations(
        self,
        train_no,
        journey_date,
        station_sequence,
        horizon=4,
    ):

        journey = self.df[
            (self.df["train_no"].astype(str) == str(train_no))
            &
            (self.df["date"] == journey_date)
        ].sort_values(
            "journey_station_index"
        )

        downstream = journey[
            journey[
                "journey_station_index"
            ] > station_sequence
        ].head(horizon)

        return downstream.to_dict(
            orient="records"
        )
