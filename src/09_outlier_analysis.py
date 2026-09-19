"""
09_outlier_analysis.py

Railway Delay Outlier & Operational Anomaly Analysis

IMPORTANT:
We DO NOT delete extreme delays.

We classify observations into:
    1. NORMAL
    2. STATISTICAL EXTREME
    3. OPERATIONAL ANOMALY
    4. POTENTIAL DATA ANOMALY

The goal is to understand whether extreme values are:
    - genuine railway disruptions
    - unusual but valid journeys
    - impossible/inconsistent data

Outputs:
    data/processed/outlier_analysis.csv
    data/processed/outlier_summary_by_train.csv
    data/processed/outlier_summary_by_station.csv
    data/processed/outlier_summary_by_section.csv
    data/processed/extreme_events.csv
"""


from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "model_data.csv"
)

OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "outlier_analysis.csv"
)

TRAIN_SUMMARY = (
    ROOT
    / "data"
    / "processed"
    / "outlier_summary_by_train.csv"
)

STATION_SUMMARY = (
    ROOT
    / "data"
    / "processed"
    / "outlier_summary_by_station.csv"
)

SECTION_SUMMARY = (
    ROOT
    / "data"
    / "processed"
    / "outlier_summary_by_section.csv"
)

EXTREME_EVENTS = (
    ROOT
    / "data"
    / "processed"
    / "extreme_events.csv"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print("RAILWAY DELAY OUTLIER & ANOMALY ANALYSIS")
print("=" * 75)

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(df["date"])

print(f"\nInput rows: {len(df):,}")


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "train_no",
        "date",
        "station_sequence"
    ]
).reset_index(drop=True)


# ============================================================
# HELPER FUNCTION
# ============================================================

def robust_zscore(series):
    """
    Robust z-score using Median Absolute Deviation.

    More resistant to extreme railway delays than
    ordinary mean/std z-score.
    """

    median = series.median()

    mad = np.median(
        np.abs(
            series.dropna() - median
        )
    )

    if mad == 0 or pd.isna(mad):
        return pd.Series(
            0.0,
            index=series.index
        )

    return (
        0.6745
        * (series - median)
        / mad
    )


# ============================================================
# GLOBAL DISTRIBUTION
# ============================================================

delay = df["delay"]

delay_change = df["delay_change"]


# ------------------------------------------------------------
# Delay percentiles
# ------------------------------------------------------------

delay_p90 = delay.quantile(0.90)
delay_p95 = delay.quantile(0.95)
delay_p99 = delay.quantile(0.99)


# ------------------------------------------------------------
# Delay-change percentiles
# ------------------------------------------------------------

abs_change = delay_change.abs()

change_p90 = abs_change.quantile(0.90)
change_p95 = abs_change.quantile(0.95)
change_p99 = abs_change.quantile(0.99)


# ============================================================
# IQR THRESHOLDS
# ============================================================

q1_delay = delay.quantile(0.25)
q3_delay = delay.quantile(0.75)

iqr_delay = q3_delay - q1_delay

delay_iqr_upper = (
    q3_delay + 1.5 * iqr_delay
)


q1_change = delay_change.quantile(0.25)
q3_change = delay_change.quantile(0.75)

iqr_change = q3_change - q1_change

change_iqr_upper = (
    q3_change + 1.5 * iqr_change
)

change_iqr_lower = (
    q1_change - 1.5 * iqr_change
)


# ============================================================
# ROBUST Z-SCORES
# ============================================================

df["delay_robust_z"] = robust_zscore(
    df["delay"]
)

df["delay_change_robust_z"] = robust_zscore(
    df["delay_change"]
)


# ============================================================
# STATISTICAL EXTREME FLAGS
# ============================================================

df["extreme_delay"] = (
    df["delay"] >= delay_p99
)


df["extreme_delay_change"] = (
    df["delay_change"].abs()
    >= change_p99
)


df["iqr_delay_outlier"] = (
    df["delay"] > delay_iqr_upper
)


df["iqr_change_outlier"] = (
    (df["delay_change"] > change_iqr_upper)
    |
    (df["delay_change"] < change_iqr_lower)
)


# ============================================================
# OPERATIONAL EXTREME
# ============================================================

# Very large delay is not automatically bad data.
#
# We explicitly call it an operational extreme.

df["operational_extreme"] = (
    (df["delay"] >= delay_p95)
    |
    (df["delay_change"].abs() >= change_p95)
)


# ============================================================
# LOCAL / SECTION ANOMALY
# ============================================================

# Compare a train-section observation against
# the normal behaviour of THAT train + section.

section_stats = (
    df
    .groupby(
        [
            "train_no",
            "section"
        ],
        dropna=False
    )
    .agg(
        section_delay_median=(
            "delay",
            "median"
        ),

        section_delay_mean=(
            "delay",
            "mean"
        ),

        section_delay_std=(
            "delay",
            "std"
        ),

        section_change_median=(
            "delay_change",
            "median"
        ),

        section_change_std=(
            "delay_change",
            "std"
        ),

        section_observations=(
            "delay",
            "count"
        )
    )
    .reset_index()
)


df = df.merge(
    section_stats,
    on=[
        "train_no",
        "section"
    ],
    how="left"
)


# ============================================================
# LOCAL DEVIATION
# ============================================================

df["delay_deviation_from_section_median"] = (
    df["delay"]
    - df["section_delay_median"]
)


df["change_deviation_from_section_median"] = (
    df["delay_change"]
    - df["section_change_median"]
)


# ============================================================
# DATA QUALITY CHECKS
# ============================================================

# These are NOT operational anomalies.
# These are potential data-quality problems.

df["possible_data_issue"] = False


# ------------------------------------------------------------
# Current delay impossible / suspicious
# ------------------------------------------------------------

df.loc[
    df["delay"].isna(),
    "possible_data_issue"
] = True


# ------------------------------------------------------------
# Missing previous delay
# ------------------------------------------------------------

# First station naturally has no previous delay.
# Therefore we DON'T classify it as bad data.

first_station = (
    df["station_sequence"] == 1
)

df.loc[
    (~first_station)
    & df["previous_delay"].isna(),
    "possible_data_issue"
] = True


# ------------------------------------------------------------
# Missing next delay
# ------------------------------------------------------------

# Last station naturally has no next delay.

last_station = (
    df["next_station"].isna()
)

df.loc[
    (~last_station)
    & df["next_delay"].isna(),
    "possible_data_issue"
] = True


# ============================================================
# EXTREME CHANGE + CURRENT DELAY
# ============================================================

df["large_delay_jump"] = (
    df["delay_change"].abs()
    >= change_p99
)


# ============================================================
# COMBINED ANOMALY TYPE
# ============================================================

df["anomaly_type"] = "NORMAL"


# ------------------------------------------------------------
# Potential data anomaly
# ------------------------------------------------------------

df.loc[
    df["possible_data_issue"],
    "anomaly_type"
] = "DATA_QUALITY"


# ------------------------------------------------------------
# Operational extreme
# ------------------------------------------------------------

df.loc[
    (~df["possible_data_issue"])
    & df["operational_extreme"],
    "anomaly_type"
] = "OPERATIONAL_EXTREME"


# ------------------------------------------------------------
# Extreme propagation event
# ------------------------------------------------------------

df.loc[
    (~df["possible_data_issue"])
    & df["large_delay_jump"],
    "anomaly_type"
] = "PROPAGATION_SPIKE"


# ------------------------------------------------------------
# Extreme both delay + change
# ------------------------------------------------------------

df.loc[
    (~df["possible_data_issue"])
    & df["extreme_delay"]
    & df["extreme_delay_change"],
    "anomaly_type"
] = "SEVERE_OPERATIONAL_EVENT"


# ============================================================
# SEVERITY
# ============================================================

df["severity"] = "NORMAL"


df.loc[
    df["operational_extreme"],
    "severity"
] = "MEDIUM"


df.loc[
    df["extreme_delay"]
    | df["extreme_delay_change"],
    "severity"
] = "HIGH"


df.loc[
    df["extreme_delay"]
    & df["extreme_delay_change"],
    "severity"
] = "CRITICAL"


# ============================================================
# ANOMALY SCORE
# ============================================================

# This is NOT a probability.
#
# It is a relative severity score for dashboard use.

delay_component = (
    df["delay_robust_z"]
    .abs()
    .clip(0, 10)
    / 10
)

change_component = (
    df["delay_change_robust_z"]
    .abs()
    .clip(0, 10)
    / 10
)


df["anomaly_score"] = (
    0.55 * delay_component
    +
    0.45 * change_component
)


df["anomaly_score"] = (
    df["anomaly_score"]
    .clip(0, 1)
)


# ============================================================
# UNCERTAINTY FLAG
# ============================================================

# Higher uncertainty when:
#
# 1. Current delay is extreme
# 2. Delay is changing rapidly
#
# This will later feed our uncertainty engine.

df["uncertainty_level"] = "LOW"


df.loc[
    df["operational_extreme"],
    "uncertainty_level"
] = "MEDIUM"


df.loc[
    df["extreme_delay"]
    | df["extreme_delay_change"],
    "uncertainty_level"
] = "HIGH"


df.loc[
    df["extreme_delay"]
    & df["extreme_delay_change"],
    "uncertainty_level"
] = "VERY_HIGH"


# ============================================================
# TRAIN SUMMARY
# ============================================================

train_summary = (
    df
    .groupby(
        "train_no",
        dropna=False
    )
    .agg(
        observations=(
            "delay",
            "count"
        ),

        mean_delay=(
            "delay",
            "mean"
        ),

        median_delay=(
            "delay",
            "median"
        ),

        p95_delay=(
            "delay",
            lambda x: x.quantile(0.95)
        ),

        p99_delay=(
            "delay",
            lambda x: x.quantile(0.99)
        ),

        max_delay=(
            "delay",
            "max"
        ),

        operational_extremes=(
            "operational_extreme",
            "sum"
        ),

        severe_events=(
            "anomaly_type",
            lambda x:
                (x == "SEVERE_OPERATIONAL_EVENT").sum()
        ),

        propagation_spikes=(
            "anomaly_type",
            lambda x:
                (x == "PROPAGATION_SPIKE").sum()
        ),

        anomaly_rate=(
            "operational_extreme",
            "mean"
        ),

        mean_anomaly_score=(
            "anomaly_score",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# STATION SUMMARY
# ============================================================

station_summary = (
    df
    .groupby(
        [
            "train_no",
            "station_code",
            "station_name"
        ],
        dropna=False
    )
    .agg(
        observations=(
            "delay",
            "count"
        ),

        median_delay=(
            "delay",
            "median"
        ),

        mean_delay=(
            "delay",
            "mean"
        ),

        p95_delay=(
            "delay",
            lambda x: x.quantile(0.95)
        ),

        max_delay=(
            "delay",
            "max"
        ),

        operational_extremes=(
            "operational_extreme",
            "sum"
        ),

        propagation_spikes=(
            "large_delay_jump",
            "sum"
        ),

        anomaly_rate=(
            "operational_extreme",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# SECTION SUMMARY
# ============================================================

section_summary = (
    df
    .groupby(
        [
            "train_no",
            "section"
        ],
        dropna=False
    )
    .agg(
        observations=(
            "delay",
            "count"
        ),

        median_delay=(
            "delay",
            "median"
        ),

        mean_delay=(
            "delay",
            "mean"
        ),

        delay_std=(
            "delay",
            "std"
        ),

        median_delay_change=(
            "delay_change",
            "median"
        ),

        delay_change_std=(
            "delay_change",
            "std"
        ),

        operational_extremes=(
            "operational_extreme",
            "sum"
        ),

        propagation_spikes=(
            "large_delay_jump",
            "sum"
        ),

        anomaly_rate=(
            "operational_extreme",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# EXTREME EVENTS
# ============================================================

extreme_events = df[
    df["anomaly_type"]
    != "NORMAL"
].copy()


extreme_events = extreme_events.sort_values(
    [
        "anomaly_score",
        "delay"
    ],
    ascending=False
)


# ============================================================
# SAVE
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    index=False
)

train_summary.to_csv(
    TRAIN_SUMMARY,
    index=False
)

station_summary.to_csv(
    STATION_SUMMARY,
    index=False
)

section_summary.to_csv(
    SECTION_SUMMARY,
    index=False
)

extreme_events.to_csv(
    EXTREME_EVENTS,
    index=False
)


# ============================================================
# CONSOLE REPORT
# ============================================================

print("\n" + "=" * 75)
print("GLOBAL THRESHOLDS")
print("=" * 75)

print(
    f"Delay P90  : {delay_p90:.2f} min"
)

print(
    f"Delay P95  : {delay_p95:.2f} min"
)

print(
    f"Delay P99  : {delay_p99:.2f} min"
)

print(
    f"|Change| P90 : {change_p90:.2f} min"
)

print(
    f"|Change| P95 : {change_p95:.2f} min"
)

print(
    f"|Change| P99 : {change_p99:.2f} min"
)


print("\n" + "=" * 75)
print("ANOMALY COUNTS")
print("=" * 75)

print(
    df["anomaly_type"]
    .value_counts()
    .to_string()
)


print("\n" + "=" * 75)
print("SEVERITY")
print("=" * 75)

print(
    df["severity"]
    .value_counts()
    .to_string()
)


print("\n" + "=" * 75)
print("UNCERTAINTY")
print("=" * 75)

print(
    df["uncertainty_level"]
    .value_counts()
    .to_string()
)


print("\n" + "=" * 75)
print("TOP 20 EXTREME EVENTS")
print("=" * 75)

columns_to_show = [
    "train_no",
    "date",
    "station_code",
    "station_name",
    "station_sequence",
    "delay",
    "delay_change",
    "section",
    "anomaly_type",
    "severity",
    "anomaly_score",
    "uncertainty_level"
]

print(
    extreme_events[
        columns_to_show
    ]
    .head(20)
    .to_string(index=False)
)


print("\n" + "=" * 75)
print("TOP TRAINS BY ANOMALY RATE")
print("=" * 75)

print(
    train_summary[
        [
            "train_no",
            "observations",
            "mean_delay",
            "median_delay",
            "p95_delay",
            "max_delay",
            "operational_extremes",
            "propagation_spikes",
            "anomaly_rate"
        ]
    ]
    .sort_values(
        "anomaly_rate",
        ascending=False
    )
    .to_string(index=False)
)


print("\n" + "=" * 75)
print("FILES CREATED")
print("=" * 75)

print(f"\nMain analysis:")
print(OUTPUT)

print("\nTrain summary:")
print(TRAIN_SUMMARY)

print("\nStation summary:")
print(STATION_SUMMARY)

print("\nSection summary:")
print(SECTION_SUMMARY)

print("\nExtreme events:")
print(EXTREME_EVENTS)

print("\n" + "=" * 75)
print("OUTLIER ANALYSIS COMPLETE")
print("=" * 75)