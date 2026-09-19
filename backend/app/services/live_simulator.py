from pathlib import Path

import pandas as pd


class LiveSimulator:

    def __init__(self):

        project_root = Path(__file__).resolve().parents[3]

        data_path = (
            project_root
            / "data"
            / "processed"
            / "model_data.csv"
        )

        self.df = pd.read_csv(data_path)

        self.df["date"] = pd.to_datetime(
            self.df["date"],
            errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        self.df["train_no"] = (
            self.df["train_no"]
            .astype(str)
        )

        self.df = self.df.sort_values(
            [
                "train_no",
                "date",
                "journey_station_index",
            ]
        ).reset_index(drop=True)

        self.active = {}


    # -----------------------------------------------------
    # JOURNEY
    # -----------------------------------------------------

    def _journey(
        self,
        train_no: str,
        journey_date: str,
    ):

        train_no = str(train_no)
        journey_date = str(journey_date)

        journey = self.df[
            (self.df["train_no"] == train_no)
            & (self.df["date"] == journey_date)
        ].copy()

        if journey.empty:

            raise ValueError(
                f"No journey found for train {train_no} "
                f"on {journey_date}"
            )

        return journey.sort_values(
            "journey_station_index"
        ).reset_index(drop=True)


    # -----------------------------------------------------
    # START
    # -----------------------------------------------------

    def start(
        self,
        train_no: str,
        journey_date: str,
    ):

        journey = self._journey(
            train_no,
            journey_date
        )

        key = (
            str(train_no),
            str(journey_date)
        )

        # Start at first meaningful station.
        index = 0

        self.active[key] = index

        return self._state_from_row(
            journey,
            index
        )


    # -----------------------------------------------------
    # TICK
    # -----------------------------------------------------

    def tick(
        self,
        train_no: str,
        journey_date: str,
    ):

        journey = self._journey(
            train_no,
            journey_date
        )

        key = (
            str(train_no),
            str(journey_date)
        )

        if key not in self.active:

            self.active[key] = 0

        current_index = self.active[key]

        if current_index < len(journey) - 1:

            current_index += 1

        self.active[key] = current_index

        return self._state_from_row(
            journey,
            current_index
        )


    # -----------------------------------------------------
    # STATE
    # -----------------------------------------------------

    def state(
        self,
        train_no: str,
        journey_date: str,
    ):

        journey = self._journey(
            train_no,
            journey_date
        )

        key = (
            str(train_no),
            str(journey_date)
        )

        if key not in self.active:

            self.active[key] = 0

        index = self.active[key]

        return self._state_from_row(
            journey,
            index
        )


    # -----------------------------------------------------
    # GET STATE (API COMPATIBILITY)
    # -----------------------------------------------------

    def get_state(
        self,
        train_no: str,
        journey_date: str,
    ):
        return self.state(
            train_no,
            journey_date
        )


    # -----------------------------------------------------
    # STATE BUILDER
    # -----------------------------------------------------

    def _state_from_row(
        self,
        journey,
        index,
    ):

        row = journey.iloc[index]

        delay = float(
            row.get("delay", 0)
            if pd.notna(row.get("delay", 0))
            else 0
        )

        previous_delay = float(
            row.get("previous_delay", delay)
            if pd.notna(row.get("previous_delay", delay))
            else delay
        )

        delay_change = float(
            row.get("delay_change", 0)
            if pd.notna(row.get("delay_change", 0))
            else 0
        )

        total_stations = len(journey)

        progress = (
            index / max(total_stations - 1, 1)
        ) * 100

        # Demo-only simulated speed.
        speed = max(
            25,
            min(
                110,
                72 - delay_change * 1.8
            )
        )

        return {
            "train_no": str(row["train_no"]),
            "journey_date": str(row["date"]),

            "station_code": str(
                row.get("station_code", "")
            ),

            "station_name": str(
                row.get("station_name", "")
            ),

            "station_sequence": int(
                row.get("station_sequence", index)
            ),

            "station_index": int(index),

            "journey_station_index": int(
                row.get("journey_station_index", index + 1)
            ),

            "delay": delay,

            "previous_delay": previous_delay,

            "delay_change": delay_change,

            "speed_kmph": round(speed, 1),

            "progress": round(progress, 1),

            "total_stations": total_stations,

            "is_finished": (
                index >= total_stations - 1
            ),
        }


    # -----------------------------------------------------
    # DOWNSTREAM STATIONS
    # -----------------------------------------------------

    def downstream(
        self,
        train_no: str,
        journey_date: str,
        horizon: int = 4,
    ):

        journey = self._journey(
            train_no,
            journey_date
        )

        key = (
            str(train_no),
            str(journey_date)
        )

        if key not in self.active:

            self.active[key] = 0

        current_index = self.active[key]

        rows = []

        end = min(
            current_index + horizon + 1,
            len(journey)
        )

        for i in range(
            current_index + 1,
            end
        ):

            row = journey.iloc[i]

            rows.append({
                "station_code": str(
                    row.get("station_code", "")
                ),

                "station_name": str(
                    row.get("station_name", "")
                ),

                "station_sequence": int(
                    row.get(
                        "station_sequence",
                        i
                    )
                ),

                "journey_station_index": int(i),
            })

        return rows