from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "processed" / "model_data.csv"

CATBOOST_MODEL = (
    ROOT / "models" / "catboost_delay_change.cbm"
)

GRU_MODEL = (
    ROOT / "models" / "gru_attention_best.pt"
)

OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "ensemble_results.csv"
)

SEQ_LEN = 6

TARGET = "next_delay_change"


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
print("CATBOOST + GRU ENSEMBLE")
print("=" * 70)

print(f"Device: {DEVICE}")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA)

df["date"] = pd.to_datetime(
    df["date"]
).dt.normalize()

df = (
    df
    .sort_values(
        [
            "date",
            "train_no",
            "journey_station_index",
        ]
    )
    .reset_index(drop=True)
)

df = df[
    df[TARGET].notna()
].copy()

print(f"Rows: {len(df):,}")


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

features = (
    numeric_features
    + categorical_features
)


# ============================================================
# NUMERIC CLEANING
# ============================================================

for col in numeric_features:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    ).fillna(0)


# ============================================================
# CANONICAL KEY
# ============================================================

# This prevents Timestamp / integer / string
# representation mismatches when joining predictions.

def make_key(
    train_no,
    date,
    station_index,
):

    return (
        str(train_no)
        + "|"
        + pd.Timestamp(date).strftime("%Y-%m-%d")
        + "|"
        + str(int(station_index))
    )


# ============================================================
# TEMPORAL JOURNEY SPLIT
# ============================================================

journeys = (
    df[
        [
            "train_no",
            "date",
        ]
    ]
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


df["split"] = "train"


df.loc[
    [
        (a, b) in val_journeys
        for a, b in zip(
            df["train_no"],
            df["date"],
        )
    ],
    "split",
] = "val"


df.loc[
    [
        (a, b) in test_journeys
        for a, b in zip(
            df["train_no"],
            df["date"],
        )
    ],
    "split",
] = "test"


test_df = df[
    df["split"] == "test"
].copy()


print(
    f"Train journeys: {len(train_journeys):,}"
)

print(
    f"Validation journeys: {len(val_journeys):,}"
)

print(
    f"Test journeys: {len(test_journeys):,}"
)

print(
    f"Test rows: {len(test_df):,}"
)


# ============================================================
# CATBOOST
# ============================================================

print("\nLoading CatBoost model...")


X_test = test_df[
    features
].copy()


for col in numeric_features:

    X_test[col] = pd.to_numeric(
        X_test[col],
        errors="coerce",
    ).fillna(0)


for col in categorical_features:

    X_test[col] = (
        X_test[col]
        .fillna("UNKNOWN")
        .astype(str)
    )


cat_model = CatBoostRegressor()

cat_model.load_model(
    CATBOOST_MODEL
)


cat_pred = cat_model.predict(
    X_test
)


print(
    f"CatBoost predictions: "
    f"{len(cat_pred):,}"
)


# ============================================================
# LOAD GRU CHECKPOINT
# ============================================================

print("\nLoading GRU checkpoint...")


checkpoint = torch.load(
    GRU_MODEL,
    map_location="cpu",
    weights_only=False,
)


print(
    f"Checkpoint type: "
    f"{type(checkpoint).__name__}"
)

print(
    f"Checkpoint tensors: "
    f"{len(checkpoint)}"
)


# ============================================================
# GRU FEATURES
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

categorical_columns = [
    "train_no",
    "station_code",
    "previous_station",
]


# ============================================================
# CATEGORICAL ENCODINGS
# ============================================================

category_maps = {}


for col in categorical_columns:

    values = (
        df[col]
        .fillna("UNKNOWN")
        .astype(str)
    )

    unique_values = sorted(
        values.unique()
    )

    mapping = {
        value: index + 1
        for index, value in enumerate(
            unique_values
        )
    }

    category_maps[col] = mapping

    df[col + "_id"] = (
        values
        .map(mapping)
        .fillna(0)
        .astype(int)
    )


# ============================================================
# CHECKPOINT VOCABULARIES
# ============================================================

train_vocab = checkpoint[
    "embeddings.0.weight"
].shape[0]

station_vocab = checkpoint[
    "embeddings.1.weight"
].shape[0]

previous_station_vocab = checkpoint[
    "embeddings.2.weight"
].shape[0]


print("\nVocabulary check:")

print(
    f"train_no             "
    f"checkpoint={train_vocab} "
    f"current={len(category_maps['train_no']) + 1}"
)

print(
    f"station_code         "
    f"checkpoint={station_vocab} "
    f"current={len(category_maps['station_code']) + 1}"
)

print(
    f"previous_station     "
    f"checkpoint={previous_station_vocab} "
    f"current={len(category_maps['previous_station']) + 1}"
)


# ============================================================
# NORMALIZATION
# ============================================================

train_rows = df[
    df["split"] == "train"
].copy()


means = train_rows[
    numeric_columns
].mean()


stds = (
    train_rows[
        numeric_columns
    ]
    .std()
    .replace(0, 1)
)


df[numeric_columns] = (
    df[numeric_columns] - means
) / stds


# ============================================================
# EXACT CHECKPOINT ARCHITECTURE
# ============================================================

embedding_0_shape = checkpoint[
    "embeddings.0.weight"
].shape

embedding_1_shape = checkpoint[
    "embeddings.1.weight"
].shape

embedding_2_shape = checkpoint[
    "embeddings.2.weight"
].shape


train_embedding_dim = (
    embedding_0_shape[1]
)

station_embedding_dim = (
    embedding_1_shape[1]
)

previous_station_embedding_dim = (
    embedding_2_shape[1]
)


gru_input_size = checkpoint[
    "gru.weight_ih_l0"
].shape[1]


# IMPORTANT:
#
# weight_hh_l0 shape:
#
# (3 * hidden_size, hidden_size)
#
# Therefore:
#
# hidden_size = second dimension

gru_hidden_size = checkpoint[
    "gru.weight_hh_l0"
].shape[1]


attention_hidden_size = checkpoint[
    "attention.0.weight"
].shape[0]


head_hidden_size = checkpoint[
    "head.0.weight"
].shape[0]


print("\nExact checkpoint architecture:")

print(
    f"Embedding 0: "
    f"{embedding_0_shape[0]} x "
    f"{embedding_0_shape[1]}"
)

print(
    f"Embedding 1: "
    f"{embedding_1_shape[0]} x "
    f"{embedding_1_shape[1]}"
)

print(
    f"Embedding 2: "
    f"{embedding_2_shape[0]} x "
    f"{embedding_2_shape[1]}"
)

print(
    f"GRU input size: "
    f"{gru_input_size}"
)

print(
    f"GRU hidden size: "
    f"{gru_hidden_size}"
)

print(
    f"Attention hidden size: "
    f"{attention_hidden_size}"
)

print(
    f"Head hidden size: "
    f"{head_hidden_size}"
)


# ============================================================
# INPUT SIZE CHECK
# ============================================================

calculated_input_size = (
    len(numeric_columns)
    + train_embedding_dim
    + station_embedding_dim
    + previous_station_embedding_dim
)


if calculated_input_size != gru_input_size:

    raise RuntimeError(
        "\nGRU input size mismatch.\n"
        f"Calculated: {calculated_input_size}\n"
        f"Checkpoint: {gru_input_size}"
    )


# ============================================================
# GRU + ATTENTION MODEL
# ============================================================

class GRUAttention(nn.Module):

    def __init__(
        self,
        train_vocab,
        station_vocab,
        previous_station_vocab,
        numeric_count,
        train_embedding_dim,
        station_embedding_dim,
        previous_station_embedding_dim,
        hidden_size,
        attention_hidden_size,
        head_hidden_size,
    ):

        super().__init__()


        self.embeddings = nn.ModuleList(
            [

                nn.Embedding(
                    train_vocab,
                    train_embedding_dim,
                ),

                nn.Embedding(
                    station_vocab,
                    station_embedding_dim,
                ),

                nn.Embedding(
                    previous_station_vocab,
                    previous_station_embedding_dim,
                ),

            ]
        )


        self.gru = nn.GRU(

            input_size=(
                numeric_count
                + train_embedding_dim
                + station_embedding_dim
                + previous_station_embedding_dim
            ),

            hidden_size=hidden_size,

            num_layers=2,

            batch_first=True,

            dropout=0.25,
        )


        self.attention = nn.Sequential(

            nn.Linear(
                hidden_size,
                attention_hidden_size,
            ),

            nn.Tanh(),

            nn.Linear(
                attention_hidden_size,
                1,
            ),
        )


        self.head = nn.Sequential(

            nn.Linear(
                hidden_size,
                head_hidden_size,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.25
            ),

            nn.Linear(
                head_hidden_size,
                1,
            ),
        )


    def forward(
        self,
        x_num,
        x_cat,
    ):

        train_id = x_cat[:, :, 0]

        station_id = x_cat[:, :, 1]

        previous_station_id = (
            x_cat[:, :, 2]
        )


        train_embedding = (
            self.embeddings[0](
                train_id
            )
        )


        station_embedding = (
            self.embeddings[1](
                station_id
            )
        )


        previous_station_embedding = (
            self.embeddings[2](
                previous_station_id
            )
        )


        x = torch.cat(
            [
                x_num,
                train_embedding,
                station_embedding,
                previous_station_embedding,
            ],
            dim=-1,
        )


        output, _ = self.gru(x)


        scores = self.attention(
            output
        )


        weights = torch.softmax(
            scores,
            dim=1,
        )


        context = torch.sum(
            output * weights,
            dim=1,
        )


        return self.head(
            context
        ).squeeze(-1)


# ============================================================
# CREATE MODEL
# ============================================================

model = GRUAttention(

    train_vocab=train_vocab,

    station_vocab=station_vocab,

    previous_station_vocab=previous_station_vocab,

    numeric_count=len(
        numeric_columns
    ),

    train_embedding_dim=(
        train_embedding_dim
    ),

    station_embedding_dim=(
        station_embedding_dim
    ),

    previous_station_embedding_dim=(
        previous_station_embedding_dim
    ),

    hidden_size=gru_hidden_size,

    attention_hidden_size=(
        attention_hidden_size
    ),

    head_hidden_size=head_hidden_size,

).to(DEVICE)


# ============================================================
# LOAD WEIGHTS
# ============================================================

print(
    "\nLoading exact GRU weights..."
)


model.load_state_dict(
    checkpoint,
    strict=True,
)


model.eval()


print(
    "✓ GRU weights loaded successfully."
)


print(
    f"GRU parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)


# ============================================================
# GRU TEST PREDICTIONS
# ============================================================

gru_predictions = []

gru_actuals = []

gru_keys = []


print(
    "\nGenerating GRU test predictions..."
)


journey_count = 0


for (
    train_no,
    date,
), journey in df.groupby(
    [
        "train_no",
        "date",
    ],
    sort=False,
):

    if (
        train_no,
        date,
    ) not in test_journeys:

        continue


    journey_count += 1


    journey = (
        journey
        .sort_values(
            "journey_station_index"
        )
        .reset_index(drop=True)
    )


    if len(journey) <= SEQ_LEN:

        continue


    for i in range(
        SEQ_LEN,
        len(journey),
    ):

        current = journey.iloc[i]

        target = current[
            TARGET
        ]


        if pd.isna(target):

            continue


        window = journey.iloc[
            i - SEQ_LEN:i
        ]


        X_num = torch.tensor(
            window[
                numeric_columns
            ].values,
            dtype=torch.float32,
        ).unsqueeze(0).to(DEVICE)


        X_cat = torch.tensor(
            window[
                [
                    "train_no_id",
                    "station_code_id",
                    "previous_station_id",
                ]
            ].values,
            dtype=torch.long,
        ).unsqueeze(0).to(DEVICE)


        with torch.no_grad():

            prediction = model(
                X_num,
                X_cat,
            ).item()


        # ----------------------------------------------------
        # CANONICAL KEY
        # ----------------------------------------------------

        key = make_key(
            train_no,
            date,
            current[
                "journey_station_index"
            ],
        )


        gru_predictions.append(
            prediction
        )

        gru_actuals.append(
            float(target)
        )

        gru_keys.append(
            key
        )


gru_pred = np.array(
    gru_predictions
)

gru_actual = np.array(
    gru_actuals
)


print(
    f"Test journeys processed: "
    f"{journey_count:,}"
)

print(
    f"GRU predictions: "
    f"{len(gru_pred):,}"
)


# ============================================================
# CATBOOST CANONICAL KEYS
# ============================================================

cat_keys = [
    make_key(
        train_no,
        date,
        station_index,
    )

    for train_no, date, station_index
    in zip(
        test_df["train_no"],
        test_df["date"],
        test_df[
            "journey_station_index"
        ],
    )
]


cat_lookup = dict(
    zip(
        cat_keys,
        cat_pred,
    )
)


# ============================================================
# ALIGN
# ============================================================

print(
    "\nAligning CatBoost + GRU predictions..."
)


aligned_cat = []

aligned_gru = []

aligned_y = []

aligned_keys = []


for key, gp, actual in zip(
    gru_keys,
    gru_pred,
    gru_actual,
):

    if key not in cat_lookup:

        continue


    aligned_cat.append(
        cat_lookup[key]
    )

    aligned_gru.append(
        gp
    )

    aligned_y.append(
        actual
    )

    aligned_keys.append(
        key
    )


aligned_cat = np.array(
    aligned_cat
)

aligned_gru = np.array(
    aligned_gru
)

aligned_y = np.array(
    aligned_y
)


print(
    f"Aligned observations: "
    f"{len(aligned_y):,}"
)


# ============================================================
# ALIGNMENT DEBUG
# ============================================================

if len(aligned_y) == 0:

    print(
        "\nDEBUG:"
    )

    print(
        "Example GRU keys:"
    )

    for key in gru_keys[:5]:

        print(
            " ",
            key
        )


    print(
        "\nExample CatBoost keys:"
    )

    for key in cat_keys[:5]:

        print(
            " ",
            key
        )


    raise RuntimeError(
        "No predictions could be aligned."
    )


# ============================================================
# INDIVIDUAL PERFORMANCE
# ============================================================

cat_mae = mean_absolute_error(
    aligned_y,
    aligned_cat,
)


cat_rmse = np.sqrt(
    mean_squared_error(
        aligned_y,
        aligned_cat,
    )
)


gru_mae = mean_absolute_error(
    aligned_y,
    aligned_gru,
)


gru_rmse = np.sqrt(
    mean_squared_error(
        aligned_y,
        aligned_gru,
    )
)


print(
    "\n" + "=" * 70
)

print(
    "INDIVIDUAL MODELS"
)

print(
    "=" * 70
)


print(
    f"CatBoost        "
    f"MAE: {cat_mae:.2f} min | "
    f"RMSE: {cat_rmse:.2f} min"
)


print(
    f"GRU + Attention "
    f"MAE: {gru_mae:.2f} min | "
    f"RMSE: {gru_rmse:.2f} min"
)


# ============================================================
# ENSEMBLE SEARCH
# ============================================================

results = []


for alpha in np.arange(
    0.0,
    1.001,
    0.05,
):

    prediction = (
        alpha * aligned_cat
        + (1.0 - alpha) * aligned_gru
    )


    mae = mean_absolute_error(
        aligned_y,
        prediction,
    )


    rmse = np.sqrt(
        mean_squared_error(
            aligned_y,
            prediction,
        )
    )


    results.append(
        {
            "catboost_weight": round(
                float(alpha),
                2,
            ),

            "gru_weight": round(
                float(1.0 - alpha),
                2,
            ),

            "MAE_minutes": mae,

            "RMSE_minutes": rmse,
        }
    )


results_df = pd.DataFrame(
    results
)


# ============================================================
# BEST MAE
# ============================================================

best_mae = results_df.loc[
    results_df[
        "MAE_minutes"
    ].idxmin()
]


# ============================================================
# BEST RMSE
# ============================================================

best_rmse = results_df.loc[
    results_df[
        "RMSE_minutes"
    ].idxmin()
]


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "ENSEMBLE RESULTS"
)

print(
    "=" * 70
)


print(
    results_df.to_string(
        index=False,
        formatters={

            "catboost_weight":
                "{:.2f}".format,

            "gru_weight":
                "{:.2f}".format,

            "MAE_minutes":
                "{:.2f}".format,

            "RMSE_minutes":
                "{:.2f}".format,
        },
    )
)


# ============================================================
# BEST ENSEMBLE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "BEST ENSEMBLE"
)

print(
    "=" * 70
)


print(
    f"Best MAE: "
    f"{best_mae['MAE_minutes']:.2f} min"
)

print(
    f"CatBoost weight: "
    f"{best_mae['catboost_weight']:.2f}"
)

print(
    f"GRU weight: "
    f"{best_mae['gru_weight']:.2f}"
)


print()


print(
    f"Best RMSE: "
    f"{best_rmse['RMSE_minutes']:.2f} min"
)

print(
    f"CatBoost weight: "
    f"{best_rmse['catboost_weight']:.2f}"
)

print(
    f"GRU weight: "
    f"{best_rmse['gru_weight']:.2f}"
)


# ============================================================
# IMPROVEMENT
# ============================================================

mae_improvement = (
    (
        cat_mae
        - best_mae["MAE_minutes"]
    )
    / cat_mae
) * 100


rmse_improvement = (
    (
        gru_rmse
        - best_rmse["RMSE_minutes"]
    )
    / gru_rmse
) * 100


print(
    "\n" + "=" * 70
)

print(
    "ENSEMBLE IMPROVEMENT"
)

print(
    "=" * 70
)


print(
    f"MAE improvement vs CatBoost: "
    f"{mae_improvement:.2f}%"
)


print(
    f"RMSE improvement vs GRU: "
    f"{rmse_improvement:.2f}%"
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT,
    index=False,
)


print(
    "\nSaved:"
)

print(
    OUTPUT
)


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL INTERPRETATION"
)

print(
    "=" * 70
)


if best_mae["MAE_minutes"] < cat_mae:

    print(
        "✓ Ensemble improves MAE over CatBoost."
    )

else:

    print(
        "→ CatBoost remains best for MAE."
    )


if best_rmse["RMSE_minutes"] < gru_rmse:

    print(
        "✓ Ensemble improves RMSE over GRU."
    )

else:

    print(
        "→ GRU + Attention remains best for RMSE."
    )


print(
    "\nTarget evaluated:"
)

print(
    "next_delay_change"
)


print(
    "\nExact passenger ETA is not being evaluated yet."
)

print(
    "That requires scheduled arrival/departure times."
)


print(
    "\nENSEMBLE EVALUATION COMPLETE."
)