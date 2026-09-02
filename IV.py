# Reusable IV / WOE / Binning Analysis Template

```python
# ================================================================
# IV / WOE / BINNING ANALYSIS TEMPLATE
# ================================================================
#
# PURPOSE:
#   1. Load your dataset
#   2. Clean numeric and categorical variables
#   3. Calculate IV for categorical variables
#   4. Calculate IV using Decision Tree bins for numeric variables
#   5. Calculate IV using Quantile (QCUT) bins for numeric variables
#   6. Keep missing values as a separate bin
#   7. Save bin-level statistics and IV comparison
#
# IMPORTANT:
#   This is an EXPLORATORY / VARIABLE SCREENING pipeline.
#   For production scorecard modelling, binning should ideally be
#   developed on TRAIN data only and then applied to validation/test.
# ================================================================


# ================================================================
# 1. IMPORT LIBRARIES
# ================================================================

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ================================================================
# 2. USER CONFIGURATION
# ================================================================

# ------------------------------------------------
# DATASET INFORMATION
# ------------------------------------------------

# >>> CHANGE THIS <<<
DATA_PATH = Path(
    r"YOUR_DATA_FOLDER\YOUR_DATASET.parquet"
)

# Output folder
# >>> CHANGE THIS IF REQUIRED <<<
OUTPUT_DIR = Path(
    r"YOUR_OUTPUT_FOLDER\IV_Binning_Output"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------
# TARGET VARIABLE
# ------------------------------------------------

# >>> CHANGE THIS <<<
TARGET_COL = "YOUR_TARGET_VARIABLE"

# Expected target values
#
# Example:
#   0 = Good
#   1 = Bad
#
# >>> CHANGE IF YOUR TARGET IS DIFFERENT <<<
GOOD_VALUE = 0
BAD_VALUE = 1


# ------------------------------------------------
# OPTIONAL ID VARIABLES
# ------------------------------------------------
#
# These variables are NOT used for IV calculation.
# Keep them here for documentation / future modelling.

ID_VARS = [
    # "CUSTOMER_ID",
    # "ACCOUNT_ID",
    # "LOAN_ID",
]


# ------------------------------------------------
# NUMERIC VARIABLES
# ------------------------------------------------
#
# >>> REPLACE THESE WITH YOUR NUMERIC VARIABLES <<<
#
# Examples:
#   AGE
#   LTV
#   INCOME
#   INTEREST_RATE
#   TENURE
#
NUMERIC_VARS = [

    # "AGE",
    # "INCOME",
    # "LTV",
    # "INTEREST_RATE",
    # "TENURE",
    # "UTILIZATION",
    # "BALANCE",

]


# ------------------------------------------------
# CATEGORICAL VARIABLES
# ------------------------------------------------
#
# >>> REPLACE THESE WITH YOUR CATEGORICAL VARIABLES <<<
#
# Examples:
#   GENDER
#   REGION
#   CUSTOMER_TYPE
#   EMPLOYMENT_TYPE
#
CATEGORICAL_VARS = [

    # "GENDER",
    # "REGION",
    # "CUSTOMER_TYPE",
    # "EMPLOYMENT_TYPE",

]


# ------------------------------------------------
# OPTIONAL NUMERIC VARIABLES
# ------------------------------------------------
#
# These are kept separate simply for your own
# classification/documentation.
#
# They will still be treated as numeric variables
# for tree/QCUT binning.
#
PROTECTED_NUMERIC_VARS = [

    # "SPECIAL_AMOUNT_1",
    # "SPECIAL_AMOUNT_2",

]


# ------------------------------------------------
# BINNING CONFIGURATION
# ------------------------------------------------

# Number of quantile bins
N_QCUT_BINS = 10

# Maximum depth of decision tree
TREE_MAX_DEPTH = 5

# Minimum observations in each tree leaf.
#
# Example:
#   0.05 = minimum 5% of observations per leaf
#
TREE_MIN_LEAF_PCT = 0.05

# Absolute minimum observations per leaf
TREE_MIN_LEAF_ABS = 20

# Random seed
RANDOM_STATE = 42


# ------------------------------------------------
# IV THRESHOLDS
# ------------------------------------------------
#
# These are only guidelines.
# They are NOT hard rules.

IV_VERY_WEAK = 0.02
IV_WEAK = 0.10
IV_MEDIUM = 0.30
IV_STRONG = 0.50


# ================================================================
# 3. BASIC VALIDATION
# ================================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

print("=" * 70)
print("IV / WOE BINNING ANALYSIS")
print("=" * 70)

print(f"Dataset : {DATA_PATH}")
print(f"Output  : {OUTPUT_DIR}")
print(f"Target  : {TARGET_COL}")


# ================================================================
# 4. LOAD ONLY REQUIRED COLUMNS
# ================================================================

print("\nReading dataset schema...")

schema_cols = pl.read_parquet(
    DATA_PATH,
    n_rows=0
).columns


# All variables required for this analysis
ALL_VARS = (
    ID_VARS
    + NUMERIC_VARS
    + CATEGORICAL_VARS
    + PROTECTED_NUMERIC_VARS
    + [TARGET_COL]
)

# Keep only columns actually present
COLS_TO_LOAD = [
    col for col in ALL_VARS
    if col in schema_cols
]


# Check missing configured columns
MISSING_CONFIGURED_COLS = [
    col for col in ALL_VARS
    if col not in schema_cols
]

if MISSING_CONFIGURED_COLS:

    print("\nWARNING: The following configured columns "
          "were NOT found in the dataset:")

    for col in MISSING_CONFIGURED_COLS:
        print(f"   - {col}")


# Target must exist
if TARGET_COL not in schema_cols:
    raise ValueError(
        f"Target variable '{TARGET_COL}' "
        f"does not exist in the dataset."
    )


print(f"\nColumns to load: {len(COLS_TO_LOAD)}")

df_pl = pl.read_parquet(
    DATA_PATH,
    columns=COLS_TO_LOAD
)

df = df_pl.to_pandas()

print(f"Rows loaded: {len(df):,}")


# ================================================================
# 5. TARGET CLEANING
# ================================================================

print("\n" + "=" * 70)
print("TARGET CHECK")
print("=" * 70)

df[TARGET_COL] = pd.to_numeric(
    df[TARGET_COL],
    errors="coerce"
)

# Remove records where target is missing
before_target_filter = len(df)

df = df[
    df[TARGET_COL].notna()
].copy()

after_target_filter = len(df)

print(
    f"Rows removed because target was missing: "
    f"{before_target_filter - after_target_filter:,}"
)


# Check target values
print("\nTarget distribution:")

print(
    df[TARGET_COL]
    .value_counts(dropna=False)
    .sort_index()
)


# ------------------------------------------------
# Validate binary target
# ------------------------------------------------

unique_target_values = set(
    df[TARGET_COL].unique()
)

expected_target_values = {
    GOOD_VALUE,
    BAD_VALUE
}

unexpected_values = (
    unique_target_values
    - expected_target_values
)

if unexpected_values:

    raise ValueError(
        "\nUnexpected target values found: "
        f"{unexpected_values}\n"
        f"Expected only: "
        f"{GOOD_VALUE} and {BAD_VALUE}"
    )


# Convert target to integer
df[TARGET_COL] = (
    df[TARGET_COL]
    .astype(int)
)

y = df[TARGET_COL]


# Target summary
total_good = (y == GOOD_VALUE).sum()
total_bad = (y == BAD_VALUE).sum()

print(f"\nGood observations : {total_good:,}")
print(f"Bad observations  : {total_bad:,}")

if len(y) > 0:
    print(
        f"Bad rate          : "
        f"{total_bad / len(y):.2%}"
    )


# ================================================================
# 6. CLEAN NUMERIC VARIABLES
# ================================================================

print("\n" + "=" * 70)
print("NUMERIC VARIABLE CLEANING")
print("=" * 70)

ALL_NUMERIC_VARS = (
    NUMERIC_VARS
    + PROTECTED_NUMERIC_VARS
)

# Only process variables actually available
AVAILABLE_NUMERIC_VARS = [
    col for col in ALL_NUMERIC_VARS
    if col in df.columns
]

for var in AVAILABLE_NUMERIC_VARS:

    df[var] = pd.to_numeric(
        df[var],
        errors="coerce"
    )

    # Convert +/- infinity to missing
    df[var] = df[var].replace(
        [np.inf, -np.inf],
        np.nan
    )

    print(
        f"{var:<35} "
        f"Missing = {df[var].isna().sum():,}"
    )


# ================================================================
# 7. CLEAN CATEGORICAL VARIABLES
# ================================================================

print("\n" + "=" * 70)
print("CATEGORICAL VARIABLE CLEANING")
print("=" * 70)

AVAILABLE_CATEGORICAL_VARS = [
    col for col in CATEGORICAL_VARS
    if col in df.columns
]

for var in AVAILABLE_CATEGORICAL_VARS:

    df[var] = (
        df[var]
        .astype(str)
        .replace({
            "": "Missing",
            "nan": "Missing",
            "None": "Missing",
            "NULL": "Missing"
        })
    )

    # Make sure actual string "NULL" is treated consistently
    df[var] = df[var].fillna("Missing")


# ================================================================
# 8. HELPER FUNCTION:
#    IV INTERPRETATION
# ================================================================

def interpret_iv(iv):

    if pd.isna(iv):
        return "Not calculated"

    if iv < IV_VERY_WEAK:
        return "Very Weak"

    elif iv < IV_WEAK:
        return "Weak"

    elif iv < IV_MEDIUM:
        return "Medium"

    elif iv < IV_STRONG:
        return "Strong"

    else:
        return "Very Strong - Investigate"


# ================================================================
# 9. HELPER FUNCTION:
#    COMPUTE BIN TABLE + WOE + IV
# ================================================================

def compute_bin_table(
    series,
    y,
    label_series=None
):

    # Nothing to calculate
    if series is None:
        return pd.DataFrame(), np.nan

    # Need at least two unique bins
    if series.nunique(dropna=False) <= 1:
        return pd.DataFrame(), np.nan


    # ------------------------------------------------
    # Create temporary dataframe
    # ------------------------------------------------

    tmp = pd.DataFrame({
        "bin": series.astype(object),
        "y": y.values
    })


    if label_series is not None:

        tmp["label"] = (
            label_series
            .astype(str)
            .values
        )


    # ------------------------------------------------
    # Aggregate
    # ------------------------------------------------

    grp = (
        tmp
        .groupby("bin", dropna=False)["y"]
        .agg(
            bad="sum",
            total="count"
        )
    )

    grp["good"] = (
        grp["total"]
        - grp["bad"]
    )


    # ------------------------------------------------
    # Overall totals
    # ------------------------------------------------

    total_bad = grp["bad"].sum()
    total_good = grp["good"].sum()


    # ------------------------------------------------
    # Distribution percentages
    # ------------------------------------------------

    grp["bad_pct"] = (
        grp["bad"]
        / total_bad
        if total_bad > 0
        else 0
    )

    grp["good_pct"] = (
        grp["good"]
        / total_good
        if total_good > 0
        else 0
    )


    # ------------------------------------------------
    # Avoid log(0)
    # ------------------------------------------------

    grp["bad_pct"] = (
        grp["bad_pct"]
        .replace(0, 1e-8)
    )

    grp["good_pct"] = (
        grp["good_pct"]
        .replace(0, 1e-8)
    )


    # ------------------------------------------------
    # Bad rate
    # ------------------------------------------------

    grp["bad_rate"] = (
        grp["bad"]
        / grp["total"]
    )


    # ------------------------------------------------
    # WOE
    # ------------------------------------------------

    grp["woe"] = np.log(
        grp["good_pct"]
        / grp["bad_pct"]
    )


    # ------------------------------------------------
    # IV contribution
    # ------------------------------------------------

    grp["iv_component"] = (
        (
            grp["good_pct"]
            - grp["bad_pct"]
        )
        * grp["woe"]
    )


    # ------------------------------------------------
    # ODR
    # ------------------------------------------------
    #
    # ODR here is effectively the bad rate.
    #
    grp["odr"] = (
        grp["bad"]
        / grp["total"]
    )


    # ------------------------------------------------
    # Reset index
    # ------------------------------------------------

    tbl = grp.reset_index()


    # ------------------------------------------------
    # Optional label
    # ------------------------------------------------

    if label_series is not None:

        mode_lbl = (
            tmp
            .groupby("bin")["label"]
            .agg(
                lambda x:
                x.mode().iloc[0]
                if not x.mode().empty
                else ""
            )
            .reset_index()
        )

        tbl = tbl.merge(
            mode_lbl,
            on="bin",
            how="left"
        )


    # ------------------------------------------------
    # Total IV
    # ------------------------------------------------

    iv = float(
        tbl["iv_component"].sum()
    )


    return tbl, iv


# ================================================================
# 10. HELPER:
#     EXTRACT TREE THRESHOLDS
# ================================================================

def extract_thresholds(model):

    thresholds = model.tree_.threshold

    thresholds = [
        float(x)
        for x in thresholds
        if (
            x != -2
            and not np.isnan(x)
        )
    ]

    return sorted(thresholds)


# ================================================================
# 11. HELPER:
#     DECISION TREE BINNING
# ================================================================

def build_tree_bins(
    df,
    var,
    y
):

    # Remove missing values for tree training
    mask = df[var].notna()

    x_nonmissing = df.loc[
        mask,
        var
    ]

    y_nonmissing = y.loc[mask]


    # Need at least 2 unique values
    if x_nonmissing.nunique() <= 1:
        return None, None, []


    # Need both target classes
    if y_nonmissing.nunique() < 2:
        return None, None, []


    # ------------------------------------------------
    # Classifier for binary target
    # ------------------------------------------------

    Model = DecisionTreeClassifier


    # ------------------------------------------------
    # Minimum leaf size
    # ------------------------------------------------

    min_leaf = max(
        TREE_MIN_LEAF_ABS,
        int(
            TREE_MIN_LEAF_PCT
            * len(x_nonmissing)
        )
    )


    # ------------------------------------------------
    # Build tree
    # ------------------------------------------------

    model = Model(
        max_depth=TREE_MAX_DEPTH,
        max_features=1,
        min_samples_leaf=min_leaf,
        random_state=RANDOM_STATE
    )


    try:

        model.fit(
            x_nonmissing.to_numpy().reshape(-1, 1),
            y_nonmissing
        )

    except Exception as e:

        print(
            f"\nTree failed for {var}: {e}"
        )

        return None, None, []


    # ------------------------------------------------
    # Extract split points
    # ------------------------------------------------

    thresholds = extract_thresholds(
        model
    )


    if len(thresholds) == 0:

        return None, model, []


    # ------------------------------------------------
    # Create bins
    # ------------------------------------------------

    bins = (
        [-np.inf]
        + thresholds
        + [np.inf]
    )


    try:

        binned = pd.cut(
            df[var],
            bins=bins,
            include_lowest=True
        )

    except Exception:

        return None, model, []


    return (
        binned,
        model,
        thresholds
    )


# ================================================================
# 12. HELPER:
#     QCUT BINNING
# ================================================================

def build_qcut_bins(
    df,
    var
):

    try:

        return pd.qcut(
            df[var],
            q=N_QCUT_BINS,
            duplicates="drop"
        )

    except Exception:

        return None


# ================================================================
# 13. OUTPUT FOLDERS
# ================================================================

TREE_DIR = OUTPUT_DIR / "Tree_Bins"
QCUT_DIR = OUTPUT_DIR / "Qcut_Bins"
CAT_DIR = OUTPUT_DIR / "Categorical_Bins"

TREE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

QCUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CAT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# 14. PROCESS CATEGORICAL VARIABLES
# ================================================================

print("\n" + "=" * 70)
print("PROCESSING CATEGORICAL VARIABLES")
print("=" * 70)

iv_results = []


for var in tqdm(
    AVAILABLE_CATEGORICAL_VARS,
    desc="Categorical variables"
):

    raw = df[var].astype(str)


    # Calculate bin statistics
    tbl, iv = compute_bin_table(
        raw,
        y,
        label_series=raw
    )


    # Save result
    output_file = (
        CAT_DIR
        / f"{var}_bins.csv"
    )

    tbl.to_csv(
        output_file,
        index=False
    )


    # Add IV result
    iv_results.append({
        "variable": var,
        "variable_type": "Categorical",
        "binning_method": "Category",
        "IV": iv,
        "IV_interpretation": interpret_iv(iv),
        "number_of_bins": (
            len(tbl)
            if not tbl.empty
            else 0
        )
    })


# ================================================================
# 15. PROCESS NUMERIC VARIABLES
# ================================================================

print("\n" + "=" * 70)
print("PROCESSING NUMERIC VARIABLES")
print("=" * 70)


for var in tqdm(
    AVAILABLE_NUMERIC_VARS,
    desc="Numeric variables"
):

    raw_vals = df[var].copy()


    # ------------------------------------------------
    # Missing mask
    # ------------------------------------------------

    missing_mask = raw_vals.isna()


    # ============================================================
    # TREE BINNING
    # ============================================================

    tree_bins, model, thresholds = (
        build_tree_bins(
            df,
            var,
            y
        )
    )


    tree_tbl = pd.DataFrame()
    tree_iv = np.nan


    if tree_bins is not None:

        # Convert to string
        tree_bins = (
            tree_bins
            .astype(object)
            .astype(str)
        )

        # Explicit Missing bin
        tree_bins[
            missing_mask
        ] = "Missing"


        tree_tbl, tree_iv = (
            compute_bin_table(
                tree_bins,
                y
            )
        )


    # Save tree bins

    tree_file = (
        TREE_DIR
        / f"{var}_tree_bins.csv"
    )

    tree_tbl.to_csv(
        tree_file,
        index=False
    )


    # Save tree thresholds

    threshold_file = (
        TREE_DIR
        / f"{var}_tree_thresholds.csv"
    )

    pd.DataFrame({
        "threshold": thresholds
    }).to_csv(
        threshold_file,
        index=False
    )


    # ============================================================
    # QCUT BINNING
    # ============================================================

    qcut_bins = build_qcut_bins(
        df,
        var
    )


    qcut_tbl = pd.DataFrame()
    qcut_iv = np.nan


    if qcut_bins is not None:

        qcut_bins = (
            qcut_bins
            .astype(object)
            .astype(str)
        )

        qcut_bins[
            missing_mask
        ] = "Missing"


        qcut_tbl, qcut_iv = (
            compute_bin_table(
                qcut_bins,
                y
            )
        )


    # Save qcut bins

    qcut_file = (
        QCUT_DIR
        / f"{var}_qcut_bins.csv"
    )

    qcut_tbl.to_csv(
        qcut_file,
        index=False
    )


    # ============================================================
    # IV RESULTS
    # ============================================================

    iv_results.append({
        "variable": var,
        "variable_type": "Numeric",
        "binning_method": "Tree",
        "IV": tree_iv,
        "IV_interpretation": interpret_iv(tree_iv),
        "number_of_bins": (
            len(tree_tbl)
            if not tree_tbl.empty
            else 0
        )
    })


    iv_results.append({
        "variable": var,
        "variable_type": "Numeric",
        "binning_method": "Qcut",
        "IV": qcut_iv,
        "IV_interpretation": interpret_iv(qcut_iv),
        "number_of_bins": (
            len(qcut_tbl)
            if not qcut_tbl.empty
            else 0
        )
    })


# ================================================================
# 16. CREATE IV SUMMARY
# ================================================================

iv_df = pd.DataFrame(
    iv_results
)


# Sort by IV
iv_df = iv_df.sort_values(
    "IV",
    ascending=False
)


# Round IV
iv_df["IV"] = (
    iv_df["IV"]
    .round(4)
)


# Save detailed IV table

iv_df.to_csv(
    OUTPUT_DIR
    / "IV_Summary_Detailed.csv",
    index=False
)


# ================================================================
# 17. CREATE VARIABLE-LEVEL IV COMPARISON
# ================================================================

# Pivot the numeric methods
iv_comparison = (
    iv_df
    .pivot_table(
        index=[
            "variable",
            "variable_type"
        ],
        columns="binning_method",
        values="IV",
        aggfunc="first"
    )
    .reset_index()
)


# Rename columns
iv_comparison = iv_comparison.rename(
    columns={
        "Category": "categorical_IV",
        "Tree": "tree_IV",
        "Qcut": "qcut_IV"
    }
)


# Add best IV
iv_cols = [
    col for col in [
        "categorical_IV",
        "tree_IV",
        "qcut_IV"
    ]
    if col in iv_comparison.columns
]


if iv_cols:

    iv_comparison["Best_IV"] = (
        iv_comparison[iv_cols]
        .max(axis=1)
    )

    iv_comparison[
        "Best_Method"
    ] = (
        iv_comparison[iv_cols]
        .idxmax(axis=1)
    )

    iv_comparison[
        "Best_IV_Interpretation"
    ] = (
        iv_comparison["Best_IV"]
        .apply(interpret_iv)
    )


# Sort
iv_comparison = (
    iv_comparison
    .sort_values(
        "Best_IV",
        ascending=False
    )
)


# Save
iv_comparison.to_csv(
    OUTPUT_DIR
    / "IV_Comparison_Table.csv",
    index=False
)


# ================================================================
# 18. CREATE VARIABLE SELECTION TABLE
# ================================================================

selection_df = (
    iv_comparison[
        [
            "variable",
            "variable_type",
            "Best_IV",
            "Best_Method",
            "Best_IV_Interpretation"
        ]
    ]
    .copy()
)


# Flag variables based on IV
selection_df["Selection_Flag"] = (
    selection_df["Best_IV"]
    .apply(
        lambda x:
        "Consider"
        if pd.notna(x) and x >= IV_VERY_WEAK
        else "Review / Exclude"
    )
)


selection_df.to_csv(
    OUTPUT_DIR
    / "Variable_Selection_Summary.csv",
    index=False
)


# ================================================================
# 19. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("PROCESSING COMPLETE")
print("=" * 70)

print(
    f"\nTotal observations analysed: "
    f"{len(df):,}"
)

print(
    f"Total variables analysed: "
    f"{len(iv_comparison):,}"
)


print("\nTop variables by IV:")

print(
    selection_df
    .head(20)
    .to_string(index=False)
)


print("\nOutput files:")

print(
    f"  {OUTPUT_DIR / 'IV_Summary_Detailed.csv'}"
)

print(
    f"  {OUTPUT_DIR / 'IV_Comparison_Table.csv'}"
)

print(
    f"  {OUTPUT_DIR / 'Variable_Selection_Summary.csv'}"
)

print(
    f"  {TREE_DIR}"
)

print(
    f"  {QCUT_DIR}"
)

print(
    f"  {CAT_DIR}"
)

print("\nDone.")
```






## TO CHANGE
# DATA_PATH = Path(
#     r"YOUR_DATA_FOLDER\YOUR_DATASET.parquet"
# )

# OUTPUT_DIR = Path(
#     r"YOUR_OUTPUT_FOLDER\IV_Binning_Output"
# )

# TARGET_COL = "YOUR_TARGET_VARIABLE"

# NUMERIC_VARS = [
#     "YOUR_NUMERIC_VARIABLE_1",
#     "YOUR_NUMERIC_VARIABLE_2",
#     "YOUR_NUMERIC_VARIABLE_3",
# ]

# CATEGORICAL_VARS = [
#     "YOUR_CATEGORICAL_VARIABLE_1",
#     "YOUR_CATEGORICAL_VARIABLE_2",
# ]

# PROTECTED_NUMERIC_VARS = [
#     # "YOUR_SPECIAL_NUMERIC_VARIABLE",
# ]
