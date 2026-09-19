from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "multihorizon_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "multihorizon_gru_attention.pt"
METRICS_PATH = MODEL_DIR / "multihorizon_metrics.json"

SEQ_LEN = 6
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 12

HORIZONS = 4

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 70)
print("MULTI-HORIZON GRU + ATTENTION")
print("=" * 70)
print(f"Device: {DEVICE}")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT)
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["train_no", "date", "journey_station_index"]
).reset_index(drop=True)

print(f"Rows: {len(df):,}")


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
        journeys.iloc[:train_end]["date"]
    )
)

val_journeys = set(
    zip(
        journeys.iloc[train_end:val_end]["train_no"],
        journeys.iloc[train_end:val_end]["date"]
    )
)

test_journeys = set(
    zip(
        journeys.iloc[val_end:]["train_no"],
        journeys.iloc[val_end:]["date"]
    )
)


def get_split(row):
    key = (row["train_no"], row["date"])

    if key in train_journeys:
        return "train"

    if key in val_journeys:
        return "val"

    if key in test_journeys:
        return "test"

    return None


df["split"] = df.apply(get_split, axis=1)

print("\nTemporal split:")
print(df["split"].value_counts())


# ============================================================
# CATEGORICAL ENCODINGS
# ============================================================

categorical_columns = [
    "train_no",
    "station_code",
    "previous_station",
]

category_maps = {}

for col in categorical_columns:

    values = df[col].fillna("UNKNOWN").astype(str)

    unique_values = sorted(values.unique())

    mapping = {
        value: index + 1
        for index, value in enumerate(unique_values)
    }

    category_maps[col] = mapping

    df[col + "_id"] = (
        values.map(mapping)
        .fillna(0)
        .astype(int)
    )


# ============================================================
# NUMERIC FEATURES
# ============================================================

numeric_columns = [
    "journey_station_index",
    "delay",
    "previous_delay",
    "delay_change",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
]


# Fill numeric features only.
# Targets are NOT filled.
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(0)


# ============================================================
# NORMALIZATION
# ============================================================

train_rows = df[df["split"] == "train"]

means = train_rows[numeric_columns].mean()
stds = train_rows[numeric_columns].std().replace(0, 1)

df[numeric_columns] = (
    df[numeric_columns] - means
) / stds


# ============================================================
# SEQUENCE DATASET
# ============================================================

class RailwaySequenceDataset(Dataset):

    def __init__(self, sequences):

        self.X_num = torch.tensor(
            np.array([x[0] for x in sequences]),
            dtype=torch.float32
        )

        self.X_cat = torch.tensor(
            np.array([x[1] for x in sequences]),
            dtype=torch.long
        )

        self.y = torch.tensor(
            np.array([x[2] for x in sequences]),
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.X_num[idx],
            self.X_cat[idx],
            self.y[idx],
        )


def build_sequences(split):

    sequences = []

    split_df = df[df["split"] == split]

    for (_, _), journey in split_df.groupby(
        ["train_no", "date"],
        sort=False
    ):

        journey = journey.sort_values(
            "journey_station_index"
        ).reset_index(drop=True)

        if len(journey) <= SEQ_LEN:
            continue

        for i in range(SEQ_LEN, len(journey)):

            current = journey.iloc[i]

            # Four future delay targets
            targets = []

            valid = True

            for h in range(1, HORIZONS + 1):

                col = f"future_delay_{h}"

                value = current[col]

                if pd.isna(value):
                    valid = False
                    break

                targets.append(float(value))

            if not valid:
                continue

            window = journey.iloc[
                i - SEQ_LEN:i
            ]

            X_num = window[numeric_columns].values

            X_cat = window[
                [
                    "train_no_id",
                    "station_code_id",
                    "previous_station_id",
                ]
            ].values

            sequences.append(
                (
                    X_num,
                    X_cat,
                    targets,
                )
            )

    return sequences


train_sequences = build_sequences("train")
val_sequences = build_sequences("val")
test_sequences = build_sequences("test")

print("\nSequence samples:")
print(f"Train: {len(train_sequences):,}")
print(f"Val:   {len(val_sequences):,}")
print(f"Test:  {len(test_sequences):,}")


train_dataset = RailwaySequenceDataset(train_sequences)
val_dataset = RailwaySequenceDataset(val_sequences)
test_dataset = RailwaySequenceDataset(test_sequences)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

class GRUAttention(nn.Module):

    def __init__(
        self,
        num_numeric,
        train_vocab,
        station_vocab,
        previous_station_vocab,
        hidden_size=128,
    ):

        super().__init__()

        self.train_embedding = nn.Embedding(
            train_vocab + 1,
            16
        )

        self.station_embedding = nn.Embedding(
            station_vocab + 1,
            32
        )

        self.previous_station_embedding = nn.Embedding(
            previous_station_vocab + 1,
            32
        )

        input_size = (
            num_numeric
            + 16
            + 32
            + 32
        )

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.25,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, HORIZONS),
        )

    def forward(self, x_num, x_cat):

        train_id = x_cat[:, :, 0]
        station_id = x_cat[:, :, 1]
        previous_station_id = x_cat[:, :, 2]

        e_train = self.train_embedding(train_id)
        e_station = self.station_embedding(station_id)
        e_previous = self.previous_station_embedding(
            previous_station_id
        )

        x = torch.cat(
            [
                x_num,
                e_train,
                e_station,
                e_previous,
            ],
            dim=-1,
        )

        output, _ = self.gru(x)

        attention_scores = self.attention(output)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1,
        )

        context = torch.sum(
            output * attention_weights,
            dim=1,
        )

        prediction = self.head(context)

        return prediction


model = GRUAttention(
    num_numeric=len(numeric_columns),
    train_vocab=len(category_maps["train_no"]),
    station_vocab=len(category_maps["station_code"]),
    previous_station_vocab=len(category_maps["previous_station"]),
).to(DEVICE)


print("\nModel parameters:")
print(
    f"{sum(p.numel() for p in model.parameters()):,}"
)


# ============================================================
# TRAINING
# ============================================================

criterion = nn.HuberLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5,
)


def evaluate(loader):

    model.eval()

    predictions = []
    actuals = []

    total_loss = 0

    with torch.no_grad():

        for X_num, X_cat, y in loader:

            X_num = X_num.to(DEVICE)
            X_cat = X_cat.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(X_num, X_cat)

            loss = criterion(pred, y)

            total_loss += loss.item()

            predictions.append(
                pred.cpu().numpy()
            )

            actuals.append(
                y.cpu().numpy()
            )

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    mae_by_horizon = []
    rmse_by_horizon = []

    for h in range(HORIZONS):

        mae = mean_absolute_error(
            actuals[:, h],
            predictions[:, h],
        )

        rmse = np.sqrt(
            mean_squared_error(
                actuals[:, h],
                predictions[:, h],
            )
        )

        mae_by_horizon.append(mae)
        rmse_by_horizon.append(rmse)

    return (
        total_loss / len(loader),
        mae_by_horizon,
        rmse_by_horizon,
    )


best_val_loss = float("inf")
patience_counter = 0

history = []


print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)


for epoch in range(1, EPOCHS + 1):

    model.train()

    train_loss = 0

    for X_num, X_cat, y in train_loader:

        X_num = X_num.to(DEVICE)
        X_cat = X_cat.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        pred = model(X_num, X_cat)

        loss = criterion(pred, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    val_loss, val_mae, val_rmse = evaluate(
        val_loader
    )

    scheduler.step(val_loss)

    avg_val_mae = np.mean(val_mae)

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": avg_val_mae,
        }
    )

    print(
        f"Epoch {epoch:03d} | "
        f"Train {train_loss:.4f} | "
        f"Val {val_loss:.4f} | "
        f"Val MAE {avg_val_mae:.2f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        patience_counter = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "numeric_columns": numeric_columns,
                "category_maps": category_maps,
                "means": means.to_dict(),
                "stds": stds.to_dict(),
                "seq_len": SEQ_LEN,
                "horizons": HORIZONS,
            },
            MODEL_PATH,
        )

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                f"\nEarly stopping at epoch {epoch}"
            )

            break


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

test_loss, test_mae, test_rmse = evaluate(
    test_loader
)


print("\n" + "=" * 70)
print("FINAL MULTI-HORIZON RESULTS")
print("=" * 70)

for h in range(HORIZONS):

    print(
        f"Horizon +{h + 1}: "
        f"MAE = {test_mae[h]:.2f} min | "
        f"RMSE = {test_rmse[h]:.2f} min"
    )


overall_mae = float(np.mean(test_mae))
overall_rmse = float(np.mean(test_rmse))

print("-" * 70)
print(
    f"Overall MAE:  {overall_mae:.2f} minutes"
)

print(
    f"Overall RMSE: {overall_rmse:.2f} minutes"
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "model": "GRU + Attention Multi-Horizon",
    "test_loss": float(test_loss),
    "overall_mae_minutes": overall_mae,
    "overall_rmse_minutes": overall_rmse,
    "horizon_mae_minutes": {
        f"+{i + 1}": float(test_mae[i])
        for i in range(HORIZONS)
    },
    "horizon_rmse_minutes": {
        f"+{i + 1}": float(test_rmse[i])
        for i in range(HORIZONS)
    },
    "sequence_length": SEQ_LEN,
    "horizons": HORIZONS,
    "device": str(DEVICE),
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


pd.DataFrame(history).to_csv(
    ROOT
    / "data"
    / "processed"
    / "multihorizon_training_history.csv",
    index=False,
)


print("\nSaved model:")
print(MODEL_PATH)

print("\nSaved metrics:")
print(METRICS_PATH)

print("\nMULTI-HORIZON TRAINING COMPLETE.")