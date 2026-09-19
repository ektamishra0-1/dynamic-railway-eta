from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.forecasting.predictor import GRUPredictor
from backend.app.providers.replay import ReplayProvider
from backend.app.services.eta_engine import ETAEngine
from backend.app.services.live_simulator import LiveSimulator


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="RailPulse AI",
    description="Dynamic ETA Forecasting Engine for Coaching Trains",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# INITIALIZE ENGINES
# ============================================================

print()
print("=" * 55)
print("🚆 RAILPULSE AI")
print("Dynamic Railway ETA Forecasting Engine")
print("=" * 55)

# Historical replay data provider
provider = ReplayProvider()

predictor = GRUPredictor()

simulator = LiveSimulator()

eta_engine = ETAEngine(
    provider=provider,
    predictor=predictor,
    simulator=simulator,
)

print()
print("=" * 55)
print("✓ Backend initialized")
print("✓ Historical Replay Provider")
print("✓ GRU + Attention PRIMARY model")
print("✓ Dynamic ETA Engine")
print("✓ Live Simulation Engine")
print("=" * 55)
print()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "name": "RailPulse AI",
        "status": "online",
        "model": "GRU + Attention",
        "mode": "Historical Replay Simulation",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "GRU + Attention",
        "primary_model": "GRU + Attention",
        "simulation_mode": True,
        "available_trains": predictor.available_trains(),
    }


# ============================================================
# TRAIN LIST
# ============================================================

@app.get("/trains")
def trains():
    return {
        "trains": predictor.available_trains()
    }


# ============================================================
# ROUTE
# ============================================================

@app.get("/journeys/{train_no}")
def journeys(train_no: str):
    try:
        df = provider.df.copy()

        journeys_df = df[
            df["train_no"].astype(str) == str(train_no)
        ].copy()

        if journeys_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No journeys found for train {train_no}",
            )

        dates = (
            journeys_df["date"]
            .dropna()
            .astype(str)
            .str[:10]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        return {
            "train_no": str(train_no),
            "dates": dates,
            "count": len(dates),
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.get("/route/{train_no}/{journey_date}")
def route(train_no: str, journey_date: str):

    try:

        df = provider.df.copy()

        journey = df[
            (df["train_no"].astype(str) == str(train_no))
            & (df["date"].astype(str) == str(journey_date))
        ].copy()

        if journey.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No journey found for {train_no} on {journey_date}",
            )

        journey = journey.sort_values(
            "journey_station_index"
        ).reset_index(drop=True)

        stations = []

        for index, (_, row) in enumerate(journey.iterrows()):

            delay = row.get("delay")

            distance = row.get("distance")

            stations.append({
                "station_code": str(
                    row.get("station_code", "")
                ),
                "station_name": str(
                    row.get("station_name", "")
                ),
                "station_sequence": int(
                    row.get("station_sequence", index)
                ),
                "journey_station_index": int(
                    row.get("journey_station_index", index)
                ),
                "delay": (
                    float(delay)
                    if delay is not None and str(delay) != "nan"
                    else None
                ),
                "distance": (
                    float(distance)
                    if distance is not None and str(distance) != "nan"
                    else None
                ),
            })

        return {
            "train_no": str(train_no),
            "journey_date": str(journey_date),
            "stations": stations,
            "total_stations": len(stations),
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================
# NORMAL FORECAST
# ============================================================

@app.get("/forecast/{train_no}/{journey_date}")
def forecast(
    train_no: str,
    journey_date: str,
    horizon: int = 4,
):

    try:

        result = eta_engine.forecast(
            train_no=train_no,
            journey_date=journey_date,
            horizon=horizon,
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


# ============================================================
# LIVE SIMULATION START
# ============================================================

@app.post("/live/start/{train_no}/{journey_date}")
def start_live(
    train_no: str,
    journey_date: str,
):

    try:

        state = simulator.start(
            train_no,
            journey_date,
        )

        return {
            **state,
            "status": "started",
            "mode": "Historical Replay Simulation",
            "state": state,
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


# ============================================================
# LIVE SIMULATION TICK
# ============================================================

@app.post("/live/tick/{train_no}/{journey_date}")
def tick_live(
    train_no: str,
    journey_date: str,
):

    try:

        state = simulator.tick(
            train_no,
            journey_date,
        )

        return {
            **state,
            "status": "updated",
            "mode": "Historical Replay Simulation",
            "state": state,
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


# ============================================================
# LIVE STATE
# ============================================================

@app.get("/live/state/{train_no}/{journey_date}")
def live_state(
    train_no: str,
    journey_date: str,
):

    try:

        state = simulator.get_state(
            train_no,
            journey_date,
        )

        if state is None:

            raise HTTPException(
                status_code=404,
                detail="No active simulation found.",
            )

        return state

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================
# LIVE FORECAST
# ============================================================

@app.get("/live/forecast/{train_no}/{journey_date}")
def live_forecast(
    train_no: str,
    journey_date: str,
    horizon: int = 4,
):

    try:

        state = simulator.get_state(
            train_no,
            journey_date,
        )

        if state is None:

            raise HTTPException(
                status_code=404,
                detail="Start the live simulation first.",
            )

        result = eta_engine.forecast_from_live(
            train_no=train_no,
            journey_date=journey_date,
            live_state=state,
            horizon=horizon,
        )

        return result

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model/info")
def model_info():

    return {
        "primary_model": "GRU + Attention",
        "secondary_model": "GRU + Attention",
        "simulation_mode": "Historical Replay",
        "available_trains": predictor.available_trains(),

        "features": predictor.NUMERIC_FEATURES + predictor.CATEGORICAL_FEATURES,
        "numeric_features": predictor.NUMERIC_FEATURES,
        "categorical_features": predictor.CATEGORICAL_FEATURES,
        "sequence_length": predictor.SEQ_LEN,
    }

