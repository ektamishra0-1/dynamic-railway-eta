import pandas as pd
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

INPUT = Path("data/processed/clean_train_data.csv")
OUTPUT = Path("data/processed/model_data.csv")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading cleaned dataset...")

df = pd.read_csv(INPUT)

print(f"Rows loaded: {len(df):,}")


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "train_no",
    "date",
    "station_code",
    "station_name",
    "delay",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")


# ============================================================
# SORT JOURNEYS
# ============================================================

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.sort_values(
    ["train_no", "date", "station_sequence"]
).reset_index(drop=True)


# ============================================================
# REBUILD JOURNEY STATION SEQUENCE
# ============================================================

print("Building sequential journey features...")

df["journey_station_index"] = (
    df.groupby(["train_no", "date"]).cumcount() + 1
)


# ============================================================
# PREVIOUS STATION / DELAY
# ============================================================

df["previous_delay"] = (
    df.groupby(["train_no", "date"])["delay"]
    .shift(1)
)

df["previous_station"] = (
    df.groupby(["train_no", "date"])["station_code"]
    .shift(1)
)


# ============================================================
# DELAY CHANGE
# ============================================================

df["delay_change"] = (
    df["delay"] - df["previous_delay"]
)


# ============================================================
# NEXT STATION / NEXT DELAY
# ============================================================

df["next_station"] = (
    df.groupby(["train_no", "date"])["station_code"]
    .shift(-1)
)

df["next_station_name"] = (
    df.groupby(["train_no", "date"])["station_name"]
    .shift(-1)
)

df["next_delay"] = (
    df.groupby(["train_no", "date"])["delay"]
    .shift(-1)
)


# ============================================================
# TARGET
# ============================================================

# What we actually want the model to learn:
#
# Given the current state of the train,
# how will the delay change at the next station?
#
# target = next_delay - current_delay

df["next_delay_change"] = (
    df["next_delay"] - df["delay"]
)


# ============================================================
# SECTION
# ============================================================

df["section"] = (
    df["previous_station"].fillna("START")
    + "_TO_"
    + df["station_code"].fillna("UNKNOWN")
)


# ============================================================
# CALENDAR FEATURES
# ============================================================

df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_month"] = df["date"].dt.day
df["month"] = df["date"].dt.month

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)


# ============================================================
# KEEP SUPERVISED TRAINING ROWS
# ============================================================

before = len(df)

model_df = df[
    df["delay"].notna()
    & df["next_delay"].notna()
].copy()

after = len(model_df)

print(f"\nRows before target filtering: {before:,}")
print(f"Rows available for modeling: {after:,}")
print(f"Rows removed: {before - after:,}")


# ============================================================
# SANITY CHECK
# ============================================================

print("\nTarget statistics:")

print(
    model_df["next_delay_change"]
    .describe()
)


# ============================================================
# SAVE
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

model_df.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MODEL DATA PREPARATION COMPLETE")
print("=" * 60)

print(f"Rows       : {len(model_df):,}")
print(f"Trains     : {model_df['train_no'].nunique()}")
print(f"Journeys   : {model_df[['train_no', 'date']].drop_duplicates().shape[0]}")
print(f"Stations   : {model_df['station_code'].nunique()}")

print(
    f"\nSaved to:\n{OUTPUT.resolve()}"
)