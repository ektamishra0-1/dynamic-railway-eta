from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("trains")
OUTPUT_DIR = Path("data")

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD ALL CSV FILES
# ============================================================

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV files found inside the 'trains' folder."
    )

print("=" * 80)
print("BUILDING CANONICAL RAILWAY DATASET")
print("=" * 80)

print(f"\nFound {len(csv_files)} CSV files.")


# ============================================================
# PROCESS EACH TRAIN
# ============================================================

all_data = []

for file in csv_files:

    print("\n" + "-" * 80)
    print(f"Processing: {file.name}")

    df = pd.read_csv(file)

    # --------------------------------------------------------
    # Standardize column names
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # Some files have DISTANCE instead of distance.
    if "distance" not in df.columns:
        print("WARNING: distance column not found.")

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required = [
        "train_no",
        "date",
        "station_code",
        "station_name",
        "delay"
    ]

    missing_columns = [
        col for col in required
        if col not in df.columns
    ]

    if missing_columns:

        print(
            f"SKIPPING FILE - missing columns: "
            f"{missing_columns}"
        )

        continue

    # --------------------------------------------------------
    # Standardize data types
    # --------------------------------------------------------

    df["train_no"] = pd.to_numeric(
        df["train_no"],
        errors="coerce"
    ).astype("Int64")

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["station_code"] = (
        df["station_code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["station_name"] = (
        df["station_name"]
        .astype(str)
        .str.strip()
    )

    df["delay"] = pd.to_numeric(
        df["delay"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Standardize distance
    # --------------------------------------------------------

    if "distance" in df.columns:

        df["distance"] = pd.to_numeric(
            df["distance"],
            errors="coerce"
        )

    else:

        # No distance available.
        df["distance"] = np.nan

    # --------------------------------------------------------
    # Add source file
    # --------------------------------------------------------

    df["source_file"] = file.name

    # --------------------------------------------------------
    # Sort by journey and station order
    # --------------------------------------------------------

    df = df.sort_values(
        ["date"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Create station sequence
    # --------------------------------------------------------

    station_order = (
        df[
            ["train_no", "station_code"]
        ]
        .drop_duplicates()
        .copy()
    )

    station_order["station_sequence"] = (
        station_order
        .groupby("train_no")
        .cumcount()
    )

    # Merge sequence back

    df = df.merge(
        station_order,
        on=[
            "train_no",
            "station_code"
        ],
        how="left"
    )

    # --------------------------------------------------------
    # Sort properly
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "train_no",
            "date",
            "station_sequence"
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Journey identifier
    # --------------------------------------------------------

    df["journey_id"] = (
        df["train_no"].astype(str)
        + "_"
        + df["date"].dt.strftime("%Y%m%d")
    )

    # --------------------------------------------------------
    # Number of stations in journey
    # --------------------------------------------------------

    df["total_stations"] = (
        df.groupby("journey_id")["station_code"]
        .transform("count")
    )

    # --------------------------------------------------------
    # Previous station information
    # --------------------------------------------------------

    df["previous_station"] = (
        df.groupby("journey_id")["station_code"]
        .shift(1)
    )

    df["previous_distance"] = (
        df.groupby("journey_id")["distance"]
        .shift(1)
    )

    df["previous_delay"] = (
        df.groupby("journey_id")["delay"]
        .shift(1)
    )

    # --------------------------------------------------------
    # Distance travelled in current section
    # --------------------------------------------------------

    df["section_distance"] = (
        df["distance"]
        - df["previous_distance"]
    )

    # --------------------------------------------------------
    # Delay change from previous station
    # --------------------------------------------------------

    df["delay_change"] = (
        df["delay"]
        - df["previous_delay"]
    )

    # --------------------------------------------------------
    # Journey progress
    # --------------------------------------------------------

    denominator = (
        df["total_stations"] - 1
    ).replace(0, np.nan)

    df["journey_progress"] = (
        df["station_sequence"]
        / denominator
    )

    # --------------------------------------------------------
    # Day-of-week features
    # --------------------------------------------------------

    df["day_of_week"] = (
        df["date"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["month"] = (
        df["date"].dt.month
    )

    # --------------------------------------------------------
    # Route identifier
    # --------------------------------------------------------

    df["route_id"] = (
        df["train_no"].astype(str)
        + "_"
        + str(
            df["station_code"].iloc[0]
        )
        + "_"
        + str(
            df["station_code"].iloc[-1]
        )
    )

    # --------------------------------------------------------
    # Keep canonical columns
    # --------------------------------------------------------

    canonical_columns = [
        "journey_id",
        "train_no",
        "date",
        "route_id",
        "station_sequence",
        "station_code",
        "station_name",
        "distance",
        "previous_station",
        "previous_distance",
        "section_distance",
        "delay",
        "previous_delay",
        "delay_change",
        "journey_progress",
        "total_stations",
        "day_of_week",
        "is_weekend",
        "month",
        "source_file"
    ]

    df = df[canonical_columns]

    all_data.append(df)

    print(
        f"Train: {df['train_no'].iloc[0]}"
    )

    print(
        f"Journeys: {df['journey_id'].nunique():,}"
    )

    print(
        f"Stations: {df['station_code'].nunique():,}"
    )

    print(
        f"Rows: {len(df):,}"
    )


# ============================================================
# COMBINE
# ============================================================

print("\n" + "=" * 80)
print("COMBINING DATASETS")
print("=" * 80)

canonical = pd.concat(
    all_data,
    ignore_index=True
)


# ============================================================
# FINAL SORT
# ============================================================

canonical = canonical.sort_values(
    [
        "train_no",
        "date",
        "station_sequence"
    ]
).reset_index(drop=True)


# ============================================================
# VALIDATION
# ============================================================

print("\nCanonical dataset shape:")
print(canonical.shape)

print("\nTrain numbers:")
print(
    canonical["train_no"]
    .dropna()
    .unique()
)

print("\nRows per train:")
print(
    canonical
    .groupby("train_no")
    .size()
)

print("\nJourneys per train:")
print(
    canonical
    .groupby("train_no")["journey_id"]
    .nunique()
)

print("\nMissing values:")
print(
    canonical.isna().sum()
    .sort_values(ascending=False)
    .head(15)
)


# ============================================================
# CHECK DUPLICATE JOURNEY-STATION OBSERVATIONS
# ============================================================

duplicate_keys = canonical.duplicated(
    subset=[
        "train_no",
        "date",
        "station_code"
    ]
).sum()

print(
    "\nDuplicate train/date/station observations:",
    duplicate_keys
)


# ============================================================
# CHECK STATION SEQUENCES
# ============================================================

print("\n" + "=" * 80)
print("CHECKING STATION SEQUENCES")
print("=" * 80)

for train_no, group in canonical.groupby("train_no"):

    sequence_counts = (
        group
        .groupby("date")["station_code"]
        .apply(tuple)
        .value_counts()
    )

    print(
        f"\nTrain {train_no}:"
    )

    print(
        f"  Unique route sequences: "
        f"{len(sequence_counts)}"
    )

    print(
        f"  Most common sequence appears "
        f"{sequence_counts.iloc[0]} times"
    )


# ============================================================
# CHECK DISTANCE MONOTONICITY
# ============================================================

print("\n" + "=" * 80)
print("CHECKING DISTANCE CONSISTENCY")
print("=" * 80)

distance_problems = 0

for journey_id, group in canonical.groupby(
    "journey_id"
):

    distances = (
        group["distance"]
        .dropna()
        .values
    )

    if len(distances) <= 1:
        continue

    differences = np.diff(distances)

    if (differences < 0).any():

        distance_problems += 1

print(
    "Journeys with decreasing distance:",
    distance_problems
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    OUTPUT_DIR
    / "canonical_train_data.csv"
)

canonical.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 80)
print("SUCCESS")
print("=" * 80)

print(
    f"\nSaved canonical dataset to:\n"
    f"{output_file.resolve()}"
)

print(
    f"\nTotal rows: {len(canonical):,}"
)

print(
    f"Total journeys: "
    f"{canonical['journey_id'].nunique():,}"
)

print(
    f"Total trains: "
    f"{canonical['train_no'].nunique():,}"
)