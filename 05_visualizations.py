from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path("data/processed/clean_train_data.csv")
OUTPUT_DIR = Path("data/visualizations")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_PATH)

# Keep only rows where delay exists for delay plots
delay_df = df.dropna(subset=["delay"]).copy()


# ============================================================
# 1. OVERALL DELAY DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    delay_df["delay"],
    bins=80
)

plt.xlabel("Delay (minutes)")
plt.ylabel("Number of observations")
plt.title("Overall Train Delay Distribution")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_overall_delay_distribution.png",
    dpi=150
)

plt.close()


# ============================================================
# 2. DELAY DISTRIBUTION WITHOUT EXTREME OUTLIERS
# ============================================================

plot_df = delay_df[
    delay_df["delay"].between(
        delay_df["delay"].quantile(0.01),
        delay_df["delay"].quantile(0.99)
    )
]

plt.figure(figsize=(10, 6))

plt.hist(
    plot_df["delay"],
    bins=60
)

plt.xlabel("Delay (minutes)")
plt.ylabel("Number of observations")
plt.title("Train Delay Distribution — 1st to 99th Percentile")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_delay_distribution_trimmed.png",
    dpi=150
)

plt.close()


# ============================================================
# 3. MEAN DELAY BY TRAIN
# ============================================================

train_delay = (
    delay_df.groupby("train_no")["delay"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(10, 6))

plt.bar(
    train_delay.index.astype(str),
    train_delay.values
)

plt.xlabel("Train number")
plt.ylabel("Mean delay (minutes)")
plt.title("Mean Delay by Train")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_mean_delay_by_train.png",
    dpi=150
)

plt.close()


# ============================================================
# 4. MEDIAN DELAY BY TRAIN
# ============================================================

train_median = (
    delay_df.groupby("train_no")["delay"]
    .median()
    .sort_values()
)

plt.figure(figsize=(10, 6))

plt.bar(
    train_median.index.astype(str),
    train_median.values
)

plt.xlabel("Train number")
plt.ylabel("Median delay (minutes)")
plt.title("Median Delay by Train")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_median_delay_by_train.png",
    dpi=150
)

plt.close()


# ============================================================
# 5. DELAY VS STATION POSITION
# ============================================================

position_delay = (
    delay_df.groupby("station_sequence")["delay"]
    .mean()
)

plt.figure(figsize=(11, 6))

plt.plot(
    position_delay.index,
    position_delay.values,
    marker="o"
)

plt.xlabel("Station position in journey")
plt.ylabel("Mean delay (minutes)")
plt.title("Mean Delay Across Journey Progress")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "05_delay_by_journey_position.png",
    dpi=150
)

plt.close()


# ============================================================
# 6. MEDIAN DELAY VS STATION POSITION
# ============================================================

position_median = (
    delay_df.groupby("station_sequence")["delay"]
    .median()
)

plt.figure(figsize=(11, 6))

plt.plot(
    position_median.index,
    position_median.values,
    marker="o"
)

plt.xlabel("Station position in journey")
plt.ylabel("Median delay (minutes)")
plt.title("Median Delay Across Journey Progress")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_median_delay_by_position.png",
    dpi=150
)

plt.close()


# ============================================================
# 7. DELAY CHANGE DISTRIBUTION
# ============================================================

change_df = df.dropna(
    subset=["delay_change"]
)

plt.figure(figsize=(10, 6))

plt.hist(
    change_df["delay_change"].clip(-120, 120),
    bins=60
)

plt.xlabel("Change in delay (minutes)")
plt.ylabel("Number of sections")
plt.title("Section-to-Section Delay Change")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "07_delay_change_distribution.png",
    dpi=150
)

plt.close()


# ============================================================
# 8. PREVIOUS DELAY VS CURRENT DELAY
# ============================================================

prop_df = df.dropna(
    subset=["previous_delay", "delay"]
).copy()

plot_prop = prop_df[
    prop_df["previous_delay"].between(-30, 300)
    & prop_df["delay"].between(-30, 300)
]

plt.figure(figsize=(8, 8))

plt.scatter(
    plot_prop["previous_delay"],
    plot_prop["delay"],
    alpha=0.15,
    s=10
)

plt.xlabel("Previous station delay (minutes)")
plt.ylabel("Current station delay (minutes)")
plt.title("Delay Propagation Between Consecutive Stations")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "08_delay_propagation.png",
    dpi=150
)

plt.close()


# ============================================================
# 9. DELAY RECOVERY VS INCREASE
# ============================================================

change = change_df["delay_change"]

categories = [
    "Recovered",
    "Unchanged",
    "Increased"
]

values = [
    (change < 0).sum(),
    (change == 0).sum(),
    (change > 0).sum()
]

plt.figure(figsize=(9, 6))

plt.bar(
    categories,
    values
)

plt.xlabel("Section behavior")
plt.ylabel("Number of sections")
plt.title("Delay Recovery vs Delay Increase")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "09_delay_recovery_vs_increase.png",
    dpi=150
)

plt.close()


# ============================================================
# 10. MISSING DELAY BY TRAIN
# ============================================================

missing_delay = (
    df.groupby("train_no")["delay"]
    .apply(lambda x: x.isna().sum())
)

plt.figure(figsize=(10, 6))

plt.bar(
    missing_delay.index.astype(str),
    missing_delay.values
)

plt.xlabel("Train number")
plt.ylabel("Missing delay observations")
plt.title("Missing Delay Values by Train")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "10_missing_delay_by_train.png",
    dpi=150
)

plt.close()


# ============================================================
# 11. JOURNEY MAXIMUM DELAY
# ============================================================

journey_max = (
    df.groupby(["train_no", "journey_id"])["delay"]
    .max()
    .dropna()
)

plt.figure(figsize=(10, 6))

plt.hist(
    journey_max,
    bins=60
)

plt.xlabel("Maximum delay during journey (minutes)")
plt.ylabel("Number of journeys")
plt.title("Maximum Delay per Journey")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "11_max_delay_per_journey.png",
    dpi=150
)

plt.close()


# ============================================================
# 12. WEEKDAY VS WEEKEND
# ============================================================

if "is_weekend" in df.columns:

    weekend_delay = (
        delay_df.groupby("is_weekend")["delay"]
        .mean()
    )

    labels = [
        "Weekday",
        "Weekend"
    ]

    values = [
        weekend_delay.get(0, 0),
        weekend_delay.get(1, 0)
    ]

    plt.figure(figsize=(8, 6))

    plt.bar(
        labels,
        values
    )

    plt.xlabel("Day type")
    plt.ylabel("Mean delay (minutes)")
    plt.title("Mean Delay: Weekday vs Weekend")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "12_weekday_vs_weekend.png",
        dpi=150
    )

    plt.close()


# ============================================================
# DONE
# ============================================================

print("=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)

print(
    f"Saved visualizations to:\n"
    f"{OUTPUT_DIR.resolve()}"
)

print(
    f"\nGenerated {len(list(OUTPUT_DIR.glob('*.png')))} plots."
)