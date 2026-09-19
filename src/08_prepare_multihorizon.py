from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "model_data.csv"
OUTPUT = ROOT / "data" / "processed" / "multihorizon_data.csv"

HORIZONS = [1, 2, 3, 4]


print("=" * 70)
print("MULTI-HORIZON DATASET PREPARATION")
print("=" * 70)

df = pd.read_csv(INPUT)

print(f"Input rows: {len(df):,}")

# ------------------------------------------------------------
# Ensure correct journey ordering
# ------------------------------------------------------------

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["train_no", "date", "journey_station_index"]
).reset_index(drop=True)


# ------------------------------------------------------------
# Create future delay targets
# ------------------------------------------------------------

group_cols = ["train_no", "date"]

for h in HORIZONS:

    df[f"future_delay_{h}"] = (
        df.groupby(group_cols)["delay"]
        .shift(-h)
    )

    df[f"future_delay_change_{h}"] = (
        df.groupby(group_cols)["delay_change"]
        .shift(-h)
    )


# ------------------------------------------------------------
# Keep useful current-state features
# ------------------------------------------------------------

feature_columns = [
    "train_no",
    "date",
    "station_code",
    "previous_station",
    "journey_station_index",
    "delay",
    "previous_delay",
    "delay_change",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
]

target_columns = []

for h in HORIZONS:
    target_columns.extend([
        f"future_delay_{h}",
        f"future_delay_change_{h}",
    ])


keep_columns = feature_columns + target_columns

df = df[keep_columns]


# ------------------------------------------------------------
# Remove rows where there is no future target
# ------------------------------------------------------------

before = len(df)

df = df.dropna(
    subset=[f"future_delay_{h}" for h in HORIZONS]
)

after = len(df)

print(f"Removed rows: {before - after:,}")
print(f"Final rows:   {after:,}")


# ------------------------------------------------------------
# Basic statistics
# ------------------------------------------------------------

print("\nTARGET COVERAGE")
print("-" * 70)

for h in HORIZONS:

    col = f"future_delay_{h}"

    print(
        f"Horizon +{h}: "
        f"{df[col].notna().sum():,} observations | "
        f"mean={df[col].mean():.2f} | "
        f"median={df[col].median():.2f}"
    )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

df.to_csv(OUTPUT, index=False)

print("\nSaved to:")
print(OUTPUT)

print("\nMULTI-HORIZON DATASET COMPLETE.")