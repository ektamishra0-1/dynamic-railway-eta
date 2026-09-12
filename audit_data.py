from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("trains")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_file(path):
    """
    Load CSV or Excel file into a pandas DataFrame.
    """

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    elif path.suffix.lower() in [".xlsx", ".xls"]:
        # Read first sheet
        return pd.read_excel(path)

    else:
        return None


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# FIND DATA FILES
# ============================================================

print_section("DATASET DISCOVERY")

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Could not find '{DATA_DIR}' folder. "
        "Make sure this script is being run from your project root."
    )

files = sorted(
    [
        f for f in DATA_DIR.iterdir()
        if f.suffix.lower() in [".csv", ".xlsx", ".xls"]
    ]
)

print(f"Data folder: {DATA_DIR.resolve()}")
print(f"Number of data files: {len(files)}")

for i, file in enumerate(files, start=1):
    print(f"{i:2}. {file.name}")


# ============================================================
# LOAD ALL FILES
# ============================================================

datasets = {}

print_section("LOADING DATASETS")

for file in files:

    try:
        df = load_file(file)

        if df is not None:
            datasets[file.name] = df

            print(
                f"\n{file.name}"
                f"\n  Rows    : {len(df):,}"
                f"\n  Columns : {len(df.columns)}"
            )

            print(f"  Columns : {list(df.columns)}")

    except Exception as e:
        print(f"\nERROR loading {file.name}")
        print(e)


# ============================================================
# BASIC DATA AUDIT
# ============================================================

print_section("BASIC DATA AUDIT")

for filename, df in datasets.items():

    print("\n" + "-" * 80)
    print(filename)

    print(f"Shape: {df.shape}")

    # --------------------------------------------------------
    # Train numbers
    # --------------------------------------------------------

    possible_train_columns = [
        col for col in df.columns
        if "train" in str(col).lower()
    ]

    if possible_train_columns:

        for col in possible_train_columns:
            unique_values = df[col].dropna().unique()

            print(
                f"\n{col}: "
                f"{len(unique_values)} unique values"
            )

            print(
                "Values:",
                unique_values[:20]
            )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    possible_date_columns = [
        col for col in df.columns
        if "date" in str(col).lower()
    ]

    if possible_date_columns:

        for col in possible_date_columns:

            dates = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            print(
                f"\nDate column: {col}"
            )

            print(
                "  Valid dates:",
                dates.notna().sum()
            )

            if dates.notna().any():
                print(
                    "  Min:",
                    dates.min()
                )

                print(
                    "  Max:",
                    dates.max()
                )

                print(
                    "  Unique dates:",
                    dates.dt.date.nunique()
                )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing values:")

    missing = df.isna().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        print("  None")

    else:

        for column, count in missing.items():

            percentage = (
                count / len(df)
            ) * 100

            print(
                f"  {column}: "
                f"{count:,} "
                f"({percentage:.2f}%)"
            )

    # --------------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    print(
        f"\nDuplicate rows: "
        f"{duplicate_count:,}"
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    if len(numeric_columns) > 0:

        print("\nNumeric columns:")

        for column in numeric_columns:

            series = df[column].dropna()

            if len(series) == 0:
                continue

            print(
                f"\n  {column}"
            )

            print(
                f"    Min     : {series.min()}"
            )

            print(
                f"    Max     : {series.max()}"
            )

            print(
                f"    Mean    : {series.mean():.2f}"
            )

            print(
                f"    Median  : {series.median():.2f}"
            )

            print(
                f"    Std     : {series.std():.2f}"
            )


# ============================================================
# CSV-SPECIFIC RAILWAY AUDIT
# ============================================================

print_section("RAILWAY RUNNING-HISTORY AUDIT")

for filename, df in datasets.items():

    # Only analyze files containing station-level history
    required_columns = {
        "train_no",
        "date",
        "station_code",
        "station_name",
        "delay"
    }

    if not required_columns.issubset(df.columns):
        continue

    print("\n" + "-" * 80)
    print(filename)

    data = df.copy()

    # --------------------------------------------------------
    # Standardize date
    # --------------------------------------------------------

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Train numbers
    # --------------------------------------------------------

    print(
        "\nTrain numbers:",
        data["train_no"].dropna().unique()
    )

    # --------------------------------------------------------
    # Journeys
    # --------------------------------------------------------

    journey_count = data["date"].dt.date.nunique()

    print(
        "Unique journey dates:",
        journey_count
    )

    # --------------------------------------------------------
    # Stations
    # --------------------------------------------------------

    station_count = data["station_code"].nunique()

    print(
        "Unique stations:",
        station_count
    )

    # --------------------------------------------------------
    # Missing delays
    # --------------------------------------------------------

    missing_delay = data["delay"].isna().sum()

    print(
        f"Missing delay values: "
        f"{missing_delay:,} "
        f"({missing_delay / len(data) * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Delay statistics
    # --------------------------------------------------------

    delay = pd.to_numeric(
        data["delay"],
        errors="coerce"
    )

    print("\nDelay statistics:")

    print(
        delay.describe()
    )

    # --------------------------------------------------------
    # Negative delays
    # --------------------------------------------------------

    negative_delays = (
        (delay < 0).sum()
    )

    print(
        "\nNegative delays:",
        negative_delays
    )

    # --------------------------------------------------------
    # Station observation counts
    # --------------------------------------------------------

    station_counts = (
        data
        .groupby("station_code")
        .size()
        .sort_values()
    )

    print("\nStation observation counts:")

    print(
        station_counts.to_string()
    )

    # --------------------------------------------------------
    # Distance audit
    # --------------------------------------------------------

    if "distance" in data.columns:

        distance = pd.to_numeric(
            data["distance"],
            errors="coerce"
        )

        print("\nDistance statistics:")

        print(
            distance.describe()
        )

        print(
            "\nDistance values missing:",
            distance.isna().sum()
        )

        # Check negative distances

        negative_distance = (
            distance < 0
        ).sum()

        print(
            "Negative distances:",
            negative_distance
        )


# ============================================================
# STATION ORDER CONSISTENCY
# ============================================================

print_section("STATION ORDER CONSISTENCY")

for filename, df in datasets.items():

    required_columns = {
        "train_no",
        "date",
        "station_code"
    }

    if not required_columns.issubset(df.columns):
        continue

    print("\n" + "-" * 80)
    print(filename)

    data = df.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    # Create station sequence based on
    # the order in which rows appear.

    sequences = (
        data
        .groupby("date")["station_code"]
        .apply(list)
    )

    if len(sequences) == 0:
        print("No valid journeys found.")
        continue

    # Use first journey as reference

    reference_date = sequences.index[0]

    reference_sequence = sequences.iloc[0]

    identical = 0

    for sequence in sequences:

        if sequence == reference_sequence:
            identical += 1

    consistency = (
        identical / len(sequences)
    ) * 100

    print(
        "Reference journey:",
        reference_date.date()
    )

    print(
        "Reference station sequence:"
    )

    print(
        " -> ".join(reference_sequence)
    )

    print(
        f"\nJourneys with identical sequence: "
        f"{identical:,}/{len(sequences):,}"
        f" ({consistency:.2f}%)"
    )


# ============================================================
# SUMMARY
# ============================================================

print_section("AUDIT COMPLETE")

print(
    f"""
Files analyzed       : {len(datasets)}
Data directory       : {DATA_DIR.resolve()}

Next step:
Use the audit output to build one canonical railway
journey dataset without blindly merging incompatible files.
"""
)