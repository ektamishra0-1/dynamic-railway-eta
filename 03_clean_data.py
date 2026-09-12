from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_ROOT / "data" / "canonical_train_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "clean_train_data.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\n" + "=" * 60)
print("CLEANING RAILWAY DATASET")
print("=" * 60)

print(f"Input rows       : {len(df):,}")
print(f"Input columns    : {len(df.columns)}")


# ============================================================
# 1. STANDARDIZE TYPES
# ============================================================

numeric_columns = [
    "train_no",
    "station_sequence",
    "delay",
    "previous_delay",
    "distance",
    "previous_distance",
    "section_distance",
    "delay_change",
    "journey_progress",
    "day_of_week",
    "is_weekend",
    "month",
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# 2. REMOVE EXACT DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates()

duplicates_removed = before - len(df)


# ============================================================
# 3. SORT JOURNEY EVENTS
# ============================================================

df = df.sort_values(
    ["train_no", "journey_id", "station_sequence"]
).reset_index(drop=True)


# ============================================================
# 4. HANDLE DISTANCE
# ============================================================
# Distance was found to be unreliable for several trains.
# We DO NOT use it for modeling at this stage.
#
# Keep the original distance columns for forensic/reference
# purposes, but explicitly mark them as unusable.

df["distance_usable"] = False


# ============================================================
# 5. VALIDATE DELAY VALUES
# ============================================================
# Negative delays are legitimate possibilities:
# negative = early arrival
# zero     = on time
# positive = late
#
# Therefore we do NOT clip negative values.

invalid_delay = df["delay"].notna() & ~np.isfinite(df["delay"])

df.loc[invalid_delay, "delay"] = np.nan


# ============================================================
# 6. MISSING DELAY FLAGS
# ============================================================
# Never convert missing delay to zero.
# Instead, explicitly tell the model whether an observation exists.

df["delay_observed"] = df["delay"].notna().astype("int8")

df["previous_delay_observed"] = (
    df["previous_delay"].notna().astype("int8")
)


# ============================================================
# 7. REBUILD SAFE JOURNEY PROGRESS
# ============================================================
# This does not depend on the broken distance column.

df["route_position"] = (
    df.groupby("journey_id")["station_sequence"]
    .transform("rank", method="dense")
    - 1
)

df["route_position"] = df["route_position"].astype("int16")


# ============================================================
# 8. ROUTE COMPLETENESS
# ============================================================

journey_lengths = (
    df.groupby("journey_id")["station_sequence"]
    .max()
    .add(1)
    .rename("journey_station_count")
)

df = df.merge(
    journey_lengths,
    on="journey_id",
    how="left"
)

df["route_completion"] = (
    df["route_position"]
    / (df["journey_station_count"] - 1).replace(0, np.nan)
)

df["route_completion"] = df["route_completion"].fillna(0)


# ============================================================
# 9. CREATE A CLEAN TRAIN/ROUTE IDENTIFIER
# ============================================================

df["train_no"] = df["train_no"].astype(str)

df["route_id"] = (
    df["train_no"].astype(str)
    + "_"
    + df["station_sequence"].astype(str)
)


# ============================================================
# 10. REMOVE OBSERVATIONS WITH NO TARGET DELAY
# ============================================================
# IMPORTANT:
# We do NOT remove them from the main cleaned dataset.
#
# Missing observations may be useful for data-quality analysis.
#
# They will only be excluded later when constructing a specific
# supervised-learning target.

# Nothing removed here.


# ============================================================
# 11. FINAL COLUMN ORDER
# ============================================================

preferred_order = [
    "journey_id",
    "train_no",
    "route_id",
    "date",
    "station_sequence",
    "station_code",
    "station_name",

    "delay",
    "delay_observed",
    "previous_delay",
    "previous_delay_observed",
    "delay_change",

    "route_position",
    "journey_station_count",
    "route_completion",

    "distance",
    "previous_distance",
    "section_distance",
    "distance_usable",

    "day_of_week",
    "is_weekend",
    "month",

    "source_file",
]

existing_columns = [
    col for col in preferred_order
    if col in df.columns
]

remaining_columns = [
    col for col in df.columns
    if col not in existing_columns
]

df = df[existing_columns + remaining_columns]


# ============================================================
# 12. SAVE
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLEANING COMPLETE")
print("=" * 60)

print(f"Rows              : {len(df):,}")
print(f"Columns           : {len(df.columns)}")
print(f"Duplicates removed: {duplicates_removed:,}")
print(f"Trains            : {df['train_no'].nunique()}")
print(f"Journeys          : {df['journey_id'].nunique():,}")
print(f"Missing delays    : {df['delay'].isna().sum():,}")
print(f"Negative delays   : {(df['delay'] < 0).sum():,}")
print(f"Distance used     : NO")
print(f"\nSaved to:")
print(OUTPUT_FILE)