from pathlib import Path
import pandas as pd
import numpy as np


DATA_FILE = Path("data/canonical_train_data.csv")


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\n" + "=" * 70)
print("RAILWAY DATA FORENSIC INVESTIGATION")
print("=" * 70)

print(f"\nRows       : {len(df):,}")
print(f"Journeys   : {df['journey_id'].nunique():,}")
print(f"Trains     : {df['train_no'].nunique()}")


# ============================================================
# 1. TRAIN SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("1. TRAIN SUMMARY")
print("=" * 70)

summary = (
    df.groupby("train_no")
    .agg(
        journeys=("journey_id", "nunique"),
        stations=("station_code", "nunique"),
        observations=("station_code", "size"),
        missing_delay=("delay", lambda x: x.isna().sum()),
        missing_distance=("distance", lambda x: x.isna().sum()),
        mean_delay=("delay", "mean"),
        median_delay=("delay", "median"),
        max_delay=("delay", "max")
    )
    .round(2)
)

print(summary.to_string())


# ============================================================
# 2. DISTANCE INVESTIGATION
# ============================================================

print("\n" + "=" * 70)
print("2. DISTANCE INVESTIGATION")
print("=" * 70)

distance_results = []

for train_no, train_df in df.groupby("train_no"):

    problems = 0
    valid_journeys = 0

    for journey_id, journey in train_df.groupby("journey_id"):

        distances = journey["distance"].dropna().values

        if len(distances) < 2:
            continue

        valid_journeys += 1

        differences = np.diff(distances)

        if (differences < 0).any():
            problems += 1

    distance_results.append({
        "train": train_no,
        "valid_journeys": valid_journeys,
        "decreasing_distance": problems,
        "problem_%": (
            problems / valid_journeys * 100
            if valid_journeys
            else 0
        )
    })

distance_results = pd.DataFrame(distance_results)

print(
    distance_results
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 3. SHOW EXAMPLES OF BAD DISTANCE SEQUENCES
# ============================================================

print("\n" + "=" * 70)
print("3. EXAMPLES OF DECREASING DISTANCE")
print("=" * 70)

shown = 0

for journey_id, journey in df.groupby("journey_id"):

    journey = journey.sort_values("station_sequence")

    distances = journey["distance"].values

    if pd.isna(distances).all():
        continue

    valid = ~pd.isna(distances)

    if valid.sum() < 2:
        continue

    diffs = np.diff(distances[valid])

    if (diffs < 0).any():

        print(f"\nJourney: {journey_id}")

        print(
            journey[
                [
                    "station_sequence",
                    "station_code",
                    "station_name",
                    "distance"
                ]
            ]
            .to_string(index=False)
        )

        shown += 1

        if shown >= 3:
            break


# ============================================================
# 4. DELAY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("4. DELAY DISTRIBUTION BY TRAIN")
print("=" * 70)

delay_summary = (
    df.groupby("train_no")["delay"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        min="min",
        max="max",
        std="std"
    )
    .round(2)
)

print(delay_summary.to_string())


# ============================================================
# 5. NEGATIVE DELAYS
# ============================================================

print("\n" + "=" * 70)
print("5. NEGATIVE DELAYS")
print("=" * 70)

negative = (
    df[df["delay"] < 0]
    .groupby("train_no")
    .size()
    .rename("negative_delay_count")
)

negative_pct = (
    df.groupby("train_no")["delay"]
    .apply(lambda x: (x < 0).sum() / x.notna().sum() * 100)
    .rename("negative_delay_%")
)

negative_summary = pd.concat(
    [negative, negative_pct],
    axis=1
).round(2)

print(
    negative_summary
    .fillna(0)
    .to_string()
)


# ============================================================
# 6. EXTREME DELAYS
# ============================================================

print("\n" + "=" * 70)
print("6. EXTREME DELAY OBSERVATIONS")
print("=" * 70)

extreme = (
    df[df["delay"] >= 300]
    .groupby("train_no")
    .size()
    .rename("delay_300_plus")
)

extreme_600 = (
    df[df["delay"] >= 600]
    .groupby("train_no")
    .size()
    .rename("delay_600_plus")
)

extreme_summary = pd.concat(
    [extreme, extreme_600],
    axis=1
).fillna(0).astype(int)

print(
    extreme_summary
    .to_string()
)


# ============================================================
# 7. DELAY CHANGE
# ============================================================

print("\n" + "=" * 70)
print("7. DELAY PROPAGATION / RECOVERY")
print("=" * 70)

delay_change_summary = (
    df.groupby("train_no")["delay_change"]
    .agg(
        mean_change="mean",
        median_change="median",
        min_change="min",
        max_change="max"
    )
    .round(2)
)

print(
    delay_change_summary
    .to_string()
)


# ============================================================
# 8. STATION-LEVEL DELAY
# ============================================================

print("\n" + "=" * 70)
print("8. STATION-LEVEL DELAY PATTERNS")
print("=" * 70)

station_summary = (
    df.groupby(
        ["train_no", "station_sequence"]
    )
    .agg(
        station=("station_code", "first"),
        mean_delay=("delay", "mean"),
        median_delay=("delay", "median"),
        delay_std=("delay", "std"),
        missing=("delay", lambda x: x.isna().sum())
    )
    .reset_index()
)

for train_no in sorted(
    df["train_no"].dropna().unique()
):

    train_stations = station_summary[
        station_summary["train_no"] == train_no
    ]

    print(f"\nTrain {int(train_no)}:")

    print(
        train_stations[
            [
                "station_sequence",
                "station",
                "mean_delay",
                "median_delay",
                "delay_std",
                "missing"
            ]
        ]
        .round(2)
        .to_string(index=False)
    )


# ============================================================
# 9. CORRELATION: PREVIOUS DELAY → CURRENT DELAY
# ============================================================

print("\n" + "=" * 70)
print("9. DELAY PROPAGATION SIGNAL")
print("=" * 70)

for train_no, train_df in df.groupby("train_no"):

    valid = train_df[
        ["previous_delay", "delay"]
    ].dropna()

    if len(valid) < 10:
        continue

    correlation = (
        valid["previous_delay"]
        .corr(valid["delay"])
    )

    print(
        f"Train {int(train_no)}:"
        f" correlation = {correlation:.3f}"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("FORENSIC INVESTIGATION COMPLETE")
print("=" * 70)

print(
    "\nNo ML model has been trained."
    "\nWe are validating the data before modeling."
)