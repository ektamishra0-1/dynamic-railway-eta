from pathlib import Path
import json

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "model_data.csv"

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "catboost_delay_change.cbm"
METRICS_PATH = MODEL_DIR / "catboost_metrics.json"

RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CATBOOST RAILWAY DELAY FORECASTING")
print("=" * 70)

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["date", "train_no", "journey_station_index"]
).reset_index(drop=True)

print(f"Rows: {len(df):,}")


# ============================================================
# TARGET
# ============================================================

TARGET = "next_delay_change"

df = df[df[TARGET].notna()].copy()

print(f"Rows with target: {len(df):,}")


# ============================================================
# TEMPORAL JOURNEY SPLIT
# ============================================================

journeys = (
    df[["train_no", "date"]]
    .drop_duplicates()
    .sort_values("date")
    .reset_index(drop=True)
)

n = len(journeys)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_journeys = set(
    zip(
        journeys.iloc[:train_end]["train_no"],
        journeys.iloc[:train_end]["date"],
    )
)

val_journeys = set(
    zip(
        journeys.iloc[train_end:val_end]["train_no"],
        journeys.iloc[train_end:val_end]["date"],
    )
)

test_journeys = set(
    zip(
        journeys.iloc[val_end:]["train_no"],
        journeys.iloc[val_end:]["date"],
    )
)


def assign_split(row):

    key = (row["train_no"], row["date"])

    if key in train_journeys:
        return "train"

    if key in val_journeys:
        return "val"

    if key in test_journeys:
        return "test"

    return None


df["split"] = df.apply(assign_split, axis=1)

print("\nSplit:")
print(df["split"].value_counts())


# ============================================================
# FEATURES
# ============================================================

numeric_features = [
    "journey_station_index",
    "delay",
    "previous_delay",
    "delay_change",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
]

categorical_features = [
    "train_no",
    "station_code",
    "previous_station",
]

features = numeric_features + categorical_features

X = df[features].copy()
y = df[TARGET].copy()


# ============================================================
# CLEAN FEATURES
# ============================================================

for col in numeric_features:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

    X[col] = X[col].fillna(0)


for col in categorical_features:

    X[col] = (
        X[col]
        .fillna("UNKNOWN")
        .astype(str)
    )


# ============================================================
# SPLIT
# ============================================================

train_mask = df["split"] == "train"
val_mask = df["split"] == "val"
test_mask = df["split"] == "test"

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print("\nDataset sizes:")
print(f"Train: {len(X_train):,}")
print(f"Val:   {len(X_val):,}")
print(f"Test:  {len(X_test):,}")


cat_indices = [
    X.columns.get_loc(col)
    for col in categorical_features
]


# ============================================================
# MODEL
# ============================================================

model = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="MAE",

    iterations=1500,

    learning_rate=0.03,

    depth=8,

    l2_leaf_reg=5,

    random_seed=RANDOM_SEED,

    verbose=100,

    early_stopping_rounds=100,

    task_type="CPU",
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CATBOOST")
print("=" * 70)

model.fit(
    X_train,
    y_train,

    cat_features=cat_indices,

    eval_set=(X_val, y_val),

    use_best_model=True,
)


# ============================================================
# TEST
# ============================================================

pred = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    pred,
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred,
    )
)


print("\n" + "=" * 70)
print("CATBOOST RESULTS")
print("=" * 70)

print(f"Test MAE:  {mae:.2f} minutes")
print(f"Test RMSE: {rmse:.2f} minutes")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = model.get_feature_importance()

importance_df = (
    pd.DataFrame(
        {
            "feature": features,
            "importance": importance,
        }
    )
    .sort_values(
        "importance",
        ascending=False,
    )
)

print("\nFEATURE IMPORTANCE")
print("-" * 70)

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

model.save_model(
    MODEL_PATH
)

metrics = {
    "model": "CatBoost",
    "target": TARGET,
    "test_mae_minutes": float(mae),
    "test_rmse_minutes": float(rmse),
    "iterations": int(model.get_best_iteration()),
    "features": features,
}

with open(
    METRICS_PATH,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=2,
    )


importance_df.to_csv(
    ROOT
    / "data"
    / "processed"
    / "catboost_feature_importance.csv",
    index=False,
)


print("\nSaved model:")
print(MODEL_PATH)

print("\nSaved metrics:")
print(METRICS_PATH)

print("\nCATBOOST COMPLETE.")