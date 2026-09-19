import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIG
# ============================================================

INPUT = Path("data/processed/model_data.csv")
OUTPUT = Path("data/processed/benchmark_results.csv")

SEQ_LEN = 6


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["train_no", "date", "journey_station_index"]
).reset_index(drop=True)


# ============================================================
# TEMPORAL JOURNEY SPLIT
# ============================================================

journeys = (
    df[["train_no", "date"]]
    .drop_duplicates()
    .sort_values("date")
)

n = len(journeys)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_journeys = set(
    map(tuple, journeys.iloc[:train_end].values)
)

val_journeys = set(
    map(tuple, journeys.iloc[train_end:val_end].values)
)

test_journeys = set(
    map(tuple, journeys.iloc[val_end:].values)
)


def get_split(row):
    key = (row["train_no"], row["date"])

    if key in train_journeys:
        return "train"

    if key in val_journeys:
        return "val"

    return "test"


df["split"] = df.apply(get_split, axis=1)


# ============================================================
# TEST DATA
# ============================================================

test = df[
    (df["split"] == "test")
    & df["next_delay_change"].notna()
    & df["delay"].notna()
].copy()


print("=" * 70)
print("RAILWAY DELAY FORECASTING BENCHMARK")
print("=" * 70)

print(f"Test observations: {len(test):,}")


# ============================================================
# MODEL 1 — PERSISTENCE
# ============================================================
#
# Simplest possible assumption:
#
# Future delay = Current delay
#
# Therefore:
#
# future delay change = 0
#
# ============================================================

test["persistence_prediction"] = 0.0


# ============================================================
# MODEL 2 — TRAIN MEDIAN
# ============================================================
#
# Historical typical delay change for each train.
#
# IMPORTANT:
# calculated ONLY from training journeys.
#
# ============================================================

train = df[
    (df["split"] == "train")
    & df["next_delay_change"].notna()
].copy()

train_median = (
    train.groupby("train_no")["next_delay_change"]
    .median()
)


test["train_median_prediction"] = (
    test["train_no"]
    .map(train_median)
    .fillna(train["next_delay_change"].median())
)


# ============================================================
# MODEL 3 — SECTION MEDIAN
# ============================================================
#
# Typical delay evolution for each railway section.
#
# ============================================================

section_median = (
    train.groupby("section")["next_delay_change"]
    .median()
)

test["section_median_prediction"] = (
    test["section"]
    .map(section_median)
    .fillna(train["next_delay_change"].median())
)


# ============================================================
# MODEL 4 — STATION POSITION MEDIAN
# ============================================================

position_median = (
    train.groupby("journey_station_index")[
        "next_delay_change"
    ]
    .median()
)

test["position_median_prediction"] = (
    test["journey_station_index"]
    .map(position_median)
    .fillna(train["next_delay_change"].median())
)


# ============================================================
# EVALUATION
# ============================================================

target = test["next_delay_change"].values


def evaluate(name, prediction):

    mae = mean_absolute_error(
        target,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            target,
            prediction
        )
    )

    return {
        "model": name,
        "MAE_minutes": mae,
        "RMSE_minutes": rmse,
    }


results = []

results.append(
    evaluate(
        "Persistence",
        test["persistence_prediction"]
    )
)

results.append(
    evaluate(
        "Train Median",
        test["train_median_prediction"]
    )
)

results.append(
    evaluate(
        "Section Median",
        test["section_median_prediction"]
    )
)

results.append(
    evaluate(
        "Station Position Median",
        test["position_median_prediction"]
    )
)


# ============================================================
# ADD GRU RESULT
# ============================================================

gru_metrics_path = Path(
    "models/gru_attention_metrics.json"
)

if gru_metrics_path.exists():

    import json

    with open(gru_metrics_path) as f:
        gru = json.load(f)

    results.append({
        "model": "GRU + Attention",
        "MAE_minutes": gru["test_mae"],
        "RMSE_minutes": gru["test_rmse"],
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "MAE_minutes"
).reset_index(drop=True)


print("\n")
print("=" * 70)
print("FINAL BENCHMARK")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# RELATIVE IMPROVEMENT
# ============================================================

gru_rows = results_df[
    results_df["model"] == "GRU + Attention"
]

if len(gru_rows) > 0:

    gru_mae = gru_rows.iloc[0]["MAE_minutes"]

    persistence_mae = results_df[
        results_df["model"] == "Persistence"
    ].iloc[0]["MAE_minutes"]

    improvement = (
        (persistence_mae - gru_mae)
        / persistence_mae
        * 100
    )

    print(
        f"\nGRU improvement vs persistence: "
        f"{improvement:.2f}%"
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT,
    index=False
)

print(
    f"\nSaved to:\n{OUTPUT.resolve()}"
)

print("\nBENCHMARK COMPLETE.")