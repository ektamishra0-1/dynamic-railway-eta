import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

INPUT = Path("data/processed/model_data.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("data/processed")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SEQ_LEN = 6
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 12


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("=" * 70)
print("DYNAMIC ETA — GRU + ATTENTION")
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
print(f"Journeys: {df[['train_no', 'date']].drop_duplicates().shape[0]}")


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


# Encode categoricals globally.
# Unknown values get index 0.

cat_maps = {}

for col in categorical_features:
    values = df[col].fillna("UNKNOWN").astype(str).unique()
    cat_maps[col] = {
        value: i + 1
        for i, value in enumerate(values)
    }

    df[col + "_id"] = (
        df[col]
        .fillna("UNKNOWN")
        .astype(str)
        .map(cat_maps[col])
        .fillna(0)
        .astype(int)
    )


# ============================================================
# TEMPORAL SPLIT
# ============================================================

journeys = (
    df[["train_no", "date"]]
    .drop_duplicates()
    .sort_values("date")
)

n = len(journeys)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_journeys = set(
    map(tuple, journeys.iloc[:train_end].values)
)

val_journeys = set(
    map(tuple, journeys.iloc[train_end:val_end].values)
)

test_journeys = set(
    map(tuple, journeys.iloc[val_end:].values)
)


def journey_set(row):
    key = (row["train_no"], row["date"])

    if key in train_journeys:
        return "train"
    if key in val_journeys:
        return "val"
    return "test"


df["split"] = df.apply(journey_set, axis=1)

print("\nTemporal split:")
print(df["split"].value_counts())


# ============================================================
# NORMALIZE NUMERIC FEATURES
# TRAIN ONLY
# ============================================================

means = {}
stds = {}

for col in numeric_features:
    mean = df.loc[df["split"] == "train", col].mean()
    std = df.loc[df["split"] == "train", col].std()

    if pd.isna(std) or std == 0:
        std = 1.0

    means[col] = float(mean)
    stds[col] = float(std)

    df[col] = (
        df[col].fillna(mean) - mean
    ) / std


# ============================================================
# SEQUENCE DATASET
# ============================================================

class SequenceDataset(Dataset):

    def __init__(self, dataframe, split_name):
        self.samples = []

        data = dataframe[dataframe["split"] == split_name].copy()

        for (_, _), group in data.groupby(
            ["train_no", "date"],
            sort=False
        ):

            group = group.sort_values(
                "journey_station_index"
            ).reset_index(drop=True)

            # We need SEQ_LEN previous observations
            # plus the current observation.
            for i in range(SEQ_LEN, len(group)):

                history = group.iloc[
                    i - SEQ_LEN:i
                ]

                target_row = group.iloc[i]

                if pd.isna(target_row["next_delay_change"]):
                    continue

                numeric = history[numeric_features].values.astype(
                    np.float32
                )

                categorical = history[
                    [c + "_id" for c in categorical_features]
                ].values.astype(np.int64)

                target = np.float32(
                    target_row["next_delay_change"]
                )

                current_delay = np.float32(
                    target_row["delay"]
                )

                self.samples.append(
                    (
                        numeric,
                        categorical,
                        target,
                        current_delay,
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        numeric, categorical, target, current_delay = (
            self.samples[idx]
        )

        return (
            torch.tensor(numeric),
            torch.tensor(categorical),
            torch.tensor(target),
            torch.tensor(current_delay),
        )


train_dataset = SequenceDataset(df, "train")
val_dataset = SequenceDataset(df, "val")
test_dataset = SequenceDataset(df, "test")

print("\nSequence samples:")
print(f"Train: {len(train_dataset):,}")
print(f"Val  : {len(val_dataset):,}")
print(f"Test : {len(test_dataset):,}")


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# MODEL
# ============================================================

class GRUAttention(nn.Module):

    def __init__(
        self,
        numeric_dim,
        categorical_sizes,
        hidden_dim=128,
        num_layers=2,
        dropout=0.25,
    ):
        super().__init__()

        # Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(size + 1, 16)
            for size in categorical_sizes
        ])

        embedding_dim = 16 * len(categorical_sizes)

        input_dim = numeric_dim + embedding_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, numeric, categorical):

        embedded = []

        for i, embedding in enumerate(
            self.embeddings
        ):
            embedded.append(
                embedding(categorical[:, :, i])
            )

        x = torch.cat(
            [numeric] + embedded,
            dim=-1
        )

        gru_out, _ = self.gru(x)

        # Attention scores
        scores = self.attention(gru_out)

        weights = torch.softmax(
            scores,
            dim=1
        )

        context = torch.sum(
            weights * gru_out,
            dim=1
        )

        context = self.dropout(context)

        output = self.head(context)

        return output.squeeze(-1)


categorical_sizes = [
    len(cat_maps[col])
    for col in categorical_features
]

model = GRUAttention(
    numeric_dim=len(numeric_features),
    categorical_sizes=categorical_sizes,
).to(DEVICE)

print("\nModel:")
print(model)


# ============================================================
# TRAINING
# ============================================================

criterion = nn.HuberLoss(
    delta=1.0
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=4,
)


def run_epoch(loader, training=True):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_count = 0

    predictions = []
    actuals = []

    for numeric, categorical, target, _ in loader:

        numeric = numeric.to(DEVICE)
        categorical = categorical.to(DEVICE)
        target = target.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            prediction = model(
                numeric,
                categorical
            )

            loss = criterion(
                prediction,
                target
            )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

        batch_size = target.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_count += batch_size

        predictions.extend(
            prediction.detach()
            .cpu()
            .numpy()
        )

        actuals.extend(
            target.detach()
            .cpu()
            .numpy()
        )

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mae = np.mean(
        np.abs(predictions - actuals)
    )

    rmse = np.sqrt(
        np.mean(
            (predictions - actuals) ** 2
        )
    )

    return (
        total_loss / total_count,
        mae,
        rmse,
    )


# ============================================================
# TRAIN LOOP
# ============================================================

best_val_loss = float("inf")
patience_counter = 0
history = []

print("\nStarting training...\n")

for epoch in range(1, EPOCHS + 1):

    train_loss, train_mae, train_rmse = run_epoch(
        train_loader,
        training=True
    )

    val_loss, val_mae, val_rmse = run_epoch(
        val_loader,
        training=False
    )

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "learning_rate": current_lr,
    })

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss {train_loss:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Train MAE {train_mae:.2f} | "
        f"Val MAE {val_mae:.2f} | "
        f"Val RMSE {val_rmse:.2f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        patience_counter = 0

        torch.save(
            model.state_dict(),
            MODEL_DIR / "gru_attention_best.pt"
        )

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                f"\nEarly stopping at epoch {epoch}"
            )

            break


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_DIR / "gru_attention_best.pt",
        map_location=DEVICE,
        weights_only=True,
    )
)


# ============================================================
# TEST
# ============================================================

test_loss, test_mae, test_rmse = run_epoch(
    test_loader,
    training=False
)

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"Test Loss : {test_loss:.4f}")
print(f"Test MAE  : {test_mae:.2f} minutes")
print(f"Test RMSE : {test_rmse:.2f} minutes")


# ============================================================
# SAVE HISTORY + METRICS
# ============================================================

with open(
    MODEL_DIR / "gru_attention_metrics.json",
    "w"
) as f:

    json.dump(
        {
            "model": "GRU + Attention",
            "sequence_length": SEQ_LEN,
            "test_loss": float(test_loss),
            "test_mae": float(test_mae),
            "test_rmse": float(test_rmse),
            "device": str(DEVICE),
            "epochs_trained": len(history),
        },
        f,
        indent=2,
    )

pd.DataFrame(history).to_csv(
    OUTPUT_DIR / "gru_training_history.csv",
    index=False
)

print(
    "\nSaved:"
    "\nmodels/gru_attention_best.pt"
    "\nmodels/gru_attention_metrics.json"
    "\ndata/processed/gru_training_history.csv"
)

print("\nTRAINING COMPLETE.")