from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path("data/processed/clean_train_data.csv")
OUTPUT_DIR = Path("data/eda")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)

print(f"Rows       : {len(df):,}")
print(f"Columns    : {len(df.columns)}")
print(f"Trains     : {df['train_no'].nunique()}")
print(f"Journeys   : {df['journey_id'].nunique()}")


# ============================================================
# BASIC STRUCTURE
# ============================================================

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = (
    df.isna()
    .sum()
    .to_frame("missing_count")
)

missing["missing_pct"] = (
    missing["missing_count"] / len(df) * 100
).round(2)

missing = missing.sort_values(
    "missing_count",
    ascending=False
)

print(missing)

missing.to_csv(
    OUTPUT_DIR / "missing_values.csv"
)


# ============================================================
# NUMERIC SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("NUMERIC SUMMARY")
print("=" * 60)

numeric_summary = df.describe(
    include=[np.number]
).T

print(numeric_summary)

numeric_summary.to_csv(
    OUTPUT_DIR / "numeric_summary.csv"
)


# ============================================================
# DELAY SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DELAY SUMMARY")
print("=" * 60)

delay = df["delay"].dropna()

delay_summary = pd.Series({
    "count": delay.count(),
    "mean": delay.mean(),
    "median": delay.median(),
    "std": delay.std(),
    "min": delay.min(),
    "25%": delay.quantile(0.25),
    "50%": delay.quantile(0.50),
    "75%": delay.quantile(0.75),
    "90%": delay.quantile(0.90),
    "95%": delay.quantile(0.95),
    "99%": delay.quantile(0.99),
    "max": delay.max(),
})

print(delay_summary)

delay_summary.to_csv(
    OUTPUT_DIR / "delay_summary.csv"
)


# ============================================================
# DELAY BY TRAIN
# ============================================================

print("\n" + "=" * 60)
print("DELAY BY TRAIN")
print("=" * 60)

train_delay = (
    df.groupby("train_no")["delay"]
    .agg(
        observations="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
    )
    .round(2)
)

print(train_delay)

train_delay.to_csv(
    OUTPUT_DIR / "delay_by_train.csv"
)


# ============================================================
# DELAY BY STATION POSITION
# ============================================================

print("\n" + "=" * 60)
print("DELAY BY STATION POSITION")
print("=" * 60)

station_position = (
    df.groupby("station_sequence")["delay"]
    .agg(
        observations="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
    )
    .round(2)
)

print(station_position)

station_position.to_csv(
    OUTPUT_DIR / "delay_by_station_position.csv"
)


# ============================================================
# DELAY BY STATION
# ============================================================

# ============================================================
# DELAY BY STATION
# ============================================================

print("\n" + "=" * 60)
print("DELAY BY STATION")
print("=" * 60)

station_delay = (
    df.groupby(["station_code", "station_name"])["delay"]
    .agg(
        observations="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
    )
    .sort_values("mean", ascending=False)
    .round(2)
)

print(station_delay.head(30))

station_delay.to_csv(
    OUTPUT_DIR / "delay_by_station.csv"
)


# ============================================================
# DELAY PROPAGATION
# ============================================================

print("\n" + "=" * 60)
print("DELAY PROPAGATION")
print("=" * 60)

propagation = (
    df.dropna(subset=["previous_delay", "delay"])
    .groupby("train_no")
    .apply(
        lambda g: pd.Series({
            "correlation": g["previous_delay"].corr(g["delay"]),
            "mean_previous_delay": g["previous_delay"].mean(),
            "mean_current_delay": g["delay"].mean(),
        }),
        include_groups=False
    )
)

print(propagation.round(3))

propagation.to_csv(
    OUTPUT_DIR / "delay_propagation.csv"
)


# ============================================================
# DELAY CHANGE
# ============================================================

print("\n" + "=" * 60)
print("DELAY CHANGE BY TRAIN")
print("=" * 60)

delay_change = (
    df.groupby("train_no")["delay_change"]
    .agg(
        observations="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
    )
    .round(2)
)

print(delay_change)

delay_change.to_csv(
    OUTPUT_DIR / "delay_change_by_train.csv"
)


# ============================================================
# DELAY INCREASE / RECOVERY
# ============================================================

print("\n" + "=" * 60)
print("DELAY INCREASE / RECOVERY")
print("=" * 60)

valid_change = df["delay_change"].dropna()

recovery_pct = (
    (valid_change < 0).sum()
    / len(valid_change)
    * 100
)

increase_pct = (
    (valid_change > 0).sum()
    / len(valid_change)
    * 100
)

unchanged_pct = (
    (valid_change == 0).sum()
    / len(valid_change)
    * 100
)

print(f"Sections recovering delay : {recovery_pct:.2f}%")
print(f"Sections increasing delay : {increase_pct:.2f}%")
print(f"Sections unchanged         : {unchanged_pct:.2f}%")


# ============================================================
# NEGATIVE DELAYS
# ============================================================

print("\n" + "=" * 60)
print("NEGATIVE DELAYS")
print("=" * 60)

negative = df[df["delay"] < 0]

print(f"Negative delay observations : {len(negative):,}")
print(
    f"Percentage of observations  : "
    f"{len(negative) / df['delay'].notna().sum() * 100:.2f}%"
)

negative_by_train = (
    negative.groupby("train_no")
    .size()
    .rename("negative_delay_count")
)

print(negative_by_train)


# ============================================================
# LARGE DELAYS
# ============================================================

print("\n" + "=" * 60)
print("LARGE DELAYS")
print("=" * 60)

for threshold in [60, 120, 300, 600]:

    count = (df["delay"] > threshold).sum()

    pct = (
        count
        / df["delay"].notna().sum()
        * 100
    )

    print(
        f"> {threshold:3} min : "
        f"{count:5,} observations "
        f"({pct:.2f}%)"
    )


# ============================================================
# JOURNEY-LEVEL DELAY
# ============================================================

print("\n" + "=" * 60)
print("JOURNEY-LEVEL DELAY")
print("=" * 60)

journey_delay = (
    df.groupby(["train_no", "journey_id"])["delay"]
    .agg(
        mean_delay="mean",
        median_delay="median",
        max_delay="max",
        min_delay="min",
        observations="count",
    )
)

print(journey_delay.describe().round(2))

journey_delay.to_csv(
    OUTPUT_DIR / "journey_level_delay.csv"
)


# ============================================================
# TRAIN JOURNEY COUNT
# ============================================================

print("\n" + "=" * 60)
print("JOURNEYS PER TRAIN")
print("=" * 60)

journeys_per_train = (
    df.groupby("train_no")["journey_id"]
    .nunique()
    .sort_index()
)

print(journeys_per_train)

journeys_per_train.to_csv(
    OUTPUT_DIR / "journeys_per_train.csv"
)


# ============================================================
# STATION COVERAGE
# ============================================================

# ============================================================
# STATION COVERAGE
# ============================================================

print("\n" + "=" * 60)
print("STATION COVERAGE")
print("=" * 60)

station_coverage = (
    df.groupby(
        ["train_no", "station_sequence", "station_code", "station_name"]
    )
    .size()
    .reset_index(name="observations")
    .sort_values(
        ["train_no", "station_sequence"]
    )
)

station_coverage.to_csv(
    OUTPUT_DIR / "station_coverage.csv",
    index=False
)

print(
    f"Unique train-station positions : "
    f"{len(station_coverage):,}"
)

print("\nSample:")
print(station_coverage.head(20))


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)

print(f"EDA outputs saved to:")
print(OUTPUT_DIR.resolve())

print("\nFiles generated:")

for file in sorted(OUTPUT_DIR.glob("*.csv")):
    print(f"  ✓ {file.name}")