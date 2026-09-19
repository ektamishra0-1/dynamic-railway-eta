import math
import pandas as pd


class ETAEngine:

    def __init__(self, provider, predictor, simulator=None):
        self.provider = provider
        self.predictor = predictor
        self.simulator = simulator

    # ============================================================
    # GET DATASET
    # ============================================================

    def _get_dataframe(self):
        """
        LiveSimulator already loads the normalized model_data.csv.
        Prefer that dataframe because it contains journey_date and
        journey_station_index in the exact format used by replay.
        """

        if self.simulator is not None and hasattr(
            self.simulator, "df"
        ):
            return self.simulator.df.copy()

        if hasattr(self.provider, "df"):
            return self.provider.df.copy()

        raise RuntimeError(
            "No replay dataframe available."
        )

    # ============================================================
    # NORMAL FORECAST
    # ============================================================

    def forecast(
        self,
        train_no: str,
        journey_date: str,
        horizon: int = 4,
    ):

        if self.simulator is not None:
            live_state = self.simulator.get_state(
                train_no,
                journey_date,
            )
        else:
            live_state = self.provider.get_state(
                train_no,
                journey_date,
            )

        return self.forecast_from_live(
            train_no=train_no,
            journey_date=journey_date,
            live_state=live_state,
            horizon=horizon,
        )

    # ============================================================
    # DYNAMIC FORECAST
    # ============================================================

    def forecast_from_live(
        self,
        train_no: str,
        journey_date: str,
        live_state: dict,
        horizon: int = 4,
    ):

        train_no = str(train_no)
        journey_date = str(journey_date)

        df = self._get_dataframe()

        # --------------------------------------------------------
        # Normalize identifiers
        # --------------------------------------------------------

        df["train_no"] = (
            df["train_no"]
            .astype(str)
        )

        # The simulator's dataframe should already have
        # journey_date. Handle date as fallback.
        if "journey_date" not in df.columns:

            if "date" in df.columns:
                df["journey_date"] = (
                    pd.to_datetime(
                        df["date"]
                    )
                    .dt.strftime("%Y-%m-%d")
                )

            else:
                raise RuntimeError(
                    "Dataset has neither journey_date nor date."
                )

        df["journey_date"] = (
            pd.to_datetime(
                df["journey_date"]
            )
            .dt.strftime("%Y-%m-%d")
        )

        # --------------------------------------------------------
        # Select journey
        # --------------------------------------------------------

        journey = df[
            (df["train_no"] == train_no)
            &
            (df["journey_date"] == journey_date)
        ].copy()

        if journey.empty:
            raise RuntimeError(
                f"No journey found for train "
                f"{train_no} on {journey_date}"
            )

        journey = journey.sort_values(
            "journey_station_index"
        ).reset_index(drop=True)

        # --------------------------------------------------------
        # Current position
        # --------------------------------------------------------

        current_index = int(
            live_state.get(
                "journey_station_index",
                0,
            )
        )

        current_delay = float(
            live_state.get(
                "delay",
                0.0,
            )
            or 0.0
        )

        # --------------------------------------------------------
        # Observed history
        # --------------------------------------------------------

        observed = journey[
            journey["journey_station_index"]
            <= current_index
        ].copy()

        observed = observed.sort_values(
            "journey_station_index"
        )

        # --------------------------------------------------------
        # Future stations
        # --------------------------------------------------------

        future = journey[
            journey["journey_station_index"]
            > current_index
        ].copy()

        future = future.head(horizon)

        forecasts = []

        running_delay = current_delay

        history = observed.copy()

        # ========================================================
        # RECURSIVE FORECAST
        # ========================================================

        for step, (_, row) in enumerate(
            future.iterrows(),
            start=1,
        ):

            # ----------------------------------------------------
            # GRU + Attention prediction
            # ----------------------------------------------------

            try:

                predicted_change = float(
                    self.predictor.predict(
                        history
                    )
                )

            except Exception as e:

                print(
                    "GRU prediction warning:",
                    repr(e),
                )

                predicted_change = 0.0

            if not math.isfinite(
                predicted_change
            ):
                predicted_change = 0.0

            # ----------------------------------------------------
            # Stability damping
            # ----------------------------------------------------

            damping = (
                1.0 / math.sqrt(step)
            )

            adjusted_change = (
                predicted_change
                * damping
            )

            predicted_delay = (
                running_delay
                + adjusted_change
            )

            # ----------------------------------------------------
            # Uncertainty
            # ----------------------------------------------------

            uncertainty = (
                8.0
                + ((step - 1) * 2.0)
            )

            lower_delay = (
                predicted_delay
                - uncertainty
            )

            upper_delay = (
                predicted_delay
                + uncertainty
            )

            confidence = max(
                0.45,
                0.95 - (step * 0.08),
            )

            # ----------------------------------------------------
            # Forecast result
            # ----------------------------------------------------

            forecasts.append(
                {
                    "station_code": str(
                        row.get(
                            "station_code",
                            "",
                        )
                    ),

                    "station_name": str(
                        row.get(
                            "station_name",
                            "",
                        )
                    ),

                    "station_sequence": int(
                        row.get(
                            "station_sequence",
                            0,
                        )
                    ),

                    "journey_station_index": int(
                        row.get(
                            "journey_station_index",
                            0,
                        )
                    ),

                    "predicted_delay": round(
                        predicted_delay,
                        2,
                    ),

                    "predicted_change": round(
                        adjusted_change,
                        2,
                    ),

                    "lower_delay": round(
                        lower_delay,
                        2,
                    ),

                    "upper_delay": round(
                        upper_delay,
                        2,
                    ),

                    "confidence": round(
                        confidence,
                        2,
                    ),
                }
            )

            # ----------------------------------------------------
            # Feed prediction downstream
            # ----------------------------------------------------

            synthetic_row = row.copy()

            synthetic_row["delay"] = (
                predicted_delay
            )

            synthetic_row[
                "previous_delay"
            ] = running_delay

            synthetic_row[
                "delay_change"
            ] = adjusted_change

            history = pd.concat(
                [
                    history,
                    synthetic_row.to_frame().T,
                ],
                ignore_index=True,
            )

            running_delay = (
                predicted_delay
            )

        # ========================================================
        # RESPONSE
        # ========================================================

        return {
            "train_no": train_no,

            "journey_date": journey_date,

            "current_station": live_state.get(
                "station_name"
            ),

            "current_delay": round(
                current_delay,
                2,
            ),

            "forecast_horizon": len(
                forecasts
            ),

            "stations": forecasts,

            "model": "GRU + Attention",

            "provider": "Historical Replay",

            "explanation": (
                "GRU + Attention analyzes the recent "
                "station sequence, current delay, "
                "previous delay, delay change, train "
                "identity, and station context to "
                "estimate downstream delay propagation. "
                "Uncertainty increases for stations "
                "farther into the future."
            ),
        }
