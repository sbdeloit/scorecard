# ============================================================
# IRAC ANALYSIS - COMPLETE MODULAR PIPELINE
# ============================================================
#
# Processing  : Polars
# Excel       : Pandas + XlsxWriter
# Progress    : tqdm
#
# Outputs:
#
# 1. Max IRAC analysis
#    - Overall sheet
#    - Product-level sheets
#    - OpeningDT vintage/date filters
#    - Status count tables
#    - X+ / 30+ / 60+ / 90+ rate tables
#    - Vintage line charts
#
# 2. Roll-rate analysis
#    - Overall sheet
#    - Product-level sheets
#    - OpeningDT filters
#    - Count + roll-rate tables
#    - Roll-forward/stabilization highlighting
#
# 3. Capture / Conversion
#    - Overall sheet
#    - Product-level sheets
#    - OpeningDT filters
#
# 4. Waterfall
#    - Overall + product levels
#    - Datewise breakdown
#
# ============================================================

import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import polars as pl

from tqdm.auto import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class IRACConfig:

    # --------------------------------------------------------
    # Input / output
    # --------------------------------------------------------

    input_file: str

    output_file: str = "IRAC_Analysis_Report.xlsx"

    separator: str = "|"

    # --------------------------------------------------------
    # Core columns
    # --------------------------------------------------------

    account_col: str | None = "LON_ACCT_NBR"

    openingdt_col: str = "OPENINGDT"

    segment_col: str = "PRODUCT_TYPE"

    # --------------------------------------------------------
    # IRAC configuration
    # --------------------------------------------------------

    irac_suffix: str = "_IRAC"

    max_irac_columns: int = 10

    # --------------------------------------------------------
    # Opening date filters
    # --------------------------------------------------------

    end_dates: list[str | date | datetime] | None = None

    # --------------------------------------------------------
    # Roll rate
    # --------------------------------------------------------

    roll_buckets: tuple[int, ...] = (0, 1, 2, 3, 4)

    roll_from_buckets: tuple[int, ...] = (2, 3)

    roll_rate_gap_columns: int = 4


# ============================================================
# TIMER
# ============================================================

class Timer:

    def __init__(self, name: str):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        print(
            f"\n{'=' * 70}\n"
            f"START: {self.name}\n"
            f"{'=' * 70}"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        elapsed = time.perf_counter() - self.start_time

        print(
            f"\n{'-' * 70}\n"
            f"COMPLETED: {self.name}\n"
            f"TIME: {elapsed:,.2f} seconds\n"
            f"{'-' * 70}"
        )


# ============================================================
# READ INPUT
# ============================================================

def read_input_data(
    input_file: str,
    separator: str = "|",
) -> pl.DataFrame:

    with Timer("READ INPUT DATA"):

        print(f"File: {input_file}")

        df = pl.read_csv(
            input_file,
            separator=separator,
        )

        print(
            f"Rows    : {df.height:,}"
        )

        print(
            f"Columns : {df.width:,}"
        )

        return df


# ============================================================
# CHECK DUPLICATE COLUMNS
# ============================================================

def check_duplicate_columns(
    df: pl.DataFrame,
) -> None:

    duplicates = [
        col
        for col in set(df.columns)
        if df.columns.count(col) > 1
    ]

    if duplicates:

        raise ValueError(
            "\nDuplicate columns detected in dataframe:\n"
            f"{duplicates}"
        )

    print("Duplicate column check: PASSED")


# ============================================================
# CREATE MAX IRAC FLAGS
# ============================================================

def create_max_irac_flags(
    df: pl.DataFrame,
    irac_suffix: str = "_IRAC",
    max_columns: int = 10,
    drop_first: bool = True,
) -> tuple[pl.DataFrame, list[str]]:

    with Timer("CREATE MAX IRAC FLAGS"):

        check_duplicate_columns(df)

        # ----------------------------------------------------
        # Identify IRAC columns
        # ----------------------------------------------------

        irac_cols = [
            col
            for col in df.columns
            if col.endswith(irac_suffix)
        ]

        if not irac_cols:

            raise ValueError(
                f"No columns ending with "
                f"{irac_suffix!r} found."
            )

        irac_cols = irac_cols[:max_columns]

        print(
            f"IRAC columns used: {len(irac_cols)}"
        )

        for col in irac_cols:
            print(f"  {col}")

        if len(irac_cols) < 2:

            raise ValueError(
                "At least two IRAC columns are required."
            )

        # ----------------------------------------------------
        # Determine target columns
        # ----------------------------------------------------

        if drop_first:
            target_cols = irac_cols[1:]
        else:
            target_cols = irac_cols

        max_cols = [
            f"{col}_max"
            for col in target_cols
        ]

        # ----------------------------------------------------
        # Check collision
        # ----------------------------------------------------

        existing_columns = set(df.columns)

        collisions = [
            col
            for col in max_cols
            if col in existing_columns
        ]

        if collisions:

            raise ValueError(
                "\nThe following max columns already exist:\n"
                f"{collisions}\n\n"
                "The dataframe may already have max flags."
            )

        # ----------------------------------------------------
        # Build expressions
        # ----------------------------------------------------

        expressions = []

        for i, target_col in enumerate(
            tqdm(
                target_cols,
                desc="Creating max IRAC flags",
                unit="column",
            )
        ):

            if drop_first:

                current_cols = irac_cols[:i + 2]

            else:

                current_cols = irac_cols[:i + 1]

            expressions.append(
                pl.max_horizontal(
                    [
                        pl.col(col)
                        .cast(
                            pl.Int64,
                            strict=False,
                        )
                        for col in current_cols
                    ]
                ).alias(
                    f"{target_col}_max"
                )
            )

        # ----------------------------------------------------
        # Add flags
        # ----------------------------------------------------

        df = df.with_columns(
            expressions
        )

        print(
            f"\nCreated {len(max_cols)} max columns:"
        )

        for col in max_cols:
            print(f"  {col}")

        print(
            f"\nNew shape: {df.shape}"
        )

        return df, max_cols


# ============================================================
# PREPARE LIGHTWEIGHT ANALYSIS DATAFRAME
# ============================================================

def prepare_analysis_df(
    df: pl.DataFrame,
    max_cols: list[str],
    segment_col: str | None = None,
    openingdt_col: str | None = None,
    account_col: str | None = None,
    extra_cols: list[str] | None = None,
) -> pl.DataFrame:

    required_cols = list(max_cols)

    for col in [
        segment_col,
        openingdt_col,
        account_col,
    ]:

        if col:
            required_cols.append(col)

    if extra_cols:
        required_cols.extend(
            extra_cols
        )

    # Remove duplicates while preserving order
    required_cols = list(
        dict.fromkeys(required_cols)
    )

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:

        raise ValueError(
            "Required columns missing:\n"
            f"{missing_cols}"
        )

    result = df.select(
        required_cols
    )

    print(
        f"Analysis dataframe: "
        f"{result.height:,} rows x "
        f"{result.width:,} columns"
    )

    return result


# ============================================================
# NORMALIZE OPENINGDT
# ============================================================

def prepare_openingdt(
    df: pl.DataFrame,
    openingdt_col: str,
) -> pl.DataFrame:

    if openingdt_col not in df.columns:

        raise ValueError(
            f"{openingdt_col!r} not found."
        )

    dtype = df.schema[
        openingdt_col
    ]

    if dtype == pl.Date:

        return df

    if dtype == pl.Datetime:

        return df.with_columns(
            pl.col(openingdt_col)
            .dt.date()
            .alias(openingdt_col)
        )

    return df.with_columns(
        pl.col(openingdt_col)
        .cast(pl.String)
        .str.strptime(
            pl.Date,
            strict=False,
        )
        .alias(openingdt_col)
    )


# ============================================================
# NORMALIZE END DATES
# ============================================================

def normalize_end_dates(
    end_dates: list[str | date | datetime] | None,
) -> list[date | None]:

    if not end_dates:
        return [None]

    normalized = []

    for value in end_dates:

        if isinstance(value, datetime):

            normalized.append(
                value.date()
            )

        elif isinstance(value, date):

            normalized.append(
                value
            )

        elif isinstance(value, str):

            try:

                normalized.append(
                    date.fromisoformat(value)
                )

            except ValueError:

                raise ValueError(
                    f"Invalid date {value!r}. "
                    "Use YYYY-MM-DD."
                )

        else:

            raise TypeError(
                "Dates must be str, date, "
                "or datetime."
            )

    return list(
        dict.fromkeys(normalized)
    )


# ============================================================
# FILTER BY OPENINGDT
# ============================================================

def filter_by_openingdt(
    df: pl.DataFrame,
    openingdt_col: str,
    end_date: date | None,
) -> pl.DataFrame:

    if end_date is None:

        return df

    return df.filter(
        pl.col(openingdt_col)
        <= pl.lit(end_date)
    )


# ============================================================
# GET SEGMENTS
# ============================================================

def get_segment_values(
    df: pl.DataFrame,
    segment_col: str | None,
) -> list:

    if not segment_col:
        return [None]

    values = (
        df
        .select(segment_col)
        .unique(
            maintain_order=True
        )
        .to_series()
        .to_list()
    )

    return values


# ============================================================
# FILTER SEGMENT
# ============================================================

def filter_segment(
    df: pl.DataFrame,
    segment_col: str | None,
    segment_value,
) -> pl.DataFrame:

    if not segment_col:

        return df

    if segment_value is None:

        return df.filter(
            pl.col(segment_col)
            .is_null()
        )

    return df.filter(
        pl.col(segment_col)
        == segment_value
    )


# ============================================================
# MAX IRAC COUNTS
# ============================================================

def calculate_max_irac_counts(
    df: pl.DataFrame,
    max_cols: list[str],
) -> pl.DataFrame:

    if not max_cols:

        raise ValueError(
            "max_cols cannot be empty."
        )

    count_tables = []

    for col in tqdm(
        max_cols,
        desc="Calculating IRAC counts",
        unit="column",
    ):

        counts = (
            df
            .select(
                pl.col(col)
                .alias("value")
            )
            .group_by("value")
            .agg(
                pl.len().alias(
                    f"{col}_count"
                )
            )
        )

        count_tables.append(
            counts
        )

    result = count_tables[0]

    for table in count_tables[1:]:

        result = result.join(
            table,
            on="value",
            how="full",
            coalesce=True,
        )

    count_columns = [
        f"{col}_count"
        for col in max_cols
    ]

    result = (
        result
        .with_columns(
            pl.col(count_columns)
            .fill_null(0)
            .cast(pl.Int64)
        )
        .sort("value")
    )

    return result


# ============================================================
# VINTAGE RATE TABLE
# ============================================================

def calculate_vintage_rates(
    count_df: pl.DataFrame,
    max_cols: list[str],
) -> pl.DataFrame:

    threshold_mapping = [
        ("X+", 1),
        ("30+", 2),
        ("60+", 3),
        ("90+", 4),
    ]

    rows = []

    for label, threshold in threshold_mapping:

        row = {
            "Cut-off": label
        }

        for col in max_cols:

            count_col = (
                f"{col}_count"
            )

            numerator = (
                count_df
                .filter(
                    pl.col("value")
                    >= threshold
                )
                .select(
                    pl.col(count_col)
                    .sum()
                )
                .item()
            )

            denominator = (
                count_df
                .select(
                    pl.col(count_col)
                    .sum()
                )
                .item()
            )

            row[col] = (
                numerator / denominator
                if denominator
                else None
            )

        rows.append(row)

    return pl.DataFrame(rows)


# ============================================================
# MAX IRAC COMPLETE CALCULATION
# ============================================================

def calculate_max_irac_analysis(
    df: pl.DataFrame,
    max_cols: list[str],
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
]:

    count_df = calculate_max_irac_counts(
        df,
        max_cols,
    )

    rate_df = calculate_vintage_rates(
        count_df,
        max_cols,
    )

    return count_df, rate_df


# ============================================================
# ROLL RATE CALCULATION
# ============================================================

def generate_roll_rates(
    df: pl.DataFrame,
    target_cols: list[str],
    account_col: str | None = None,
    considered_buckets: tuple[int, ...] = (
        0, 1, 2, 3, 4
    ),
) -> pl.DataFrame:

    results = []

    for position, from_col in enumerate(
        tqdm(
            target_cols[:-1],
            desc="Generating roll rates",
            unit="snapshot",
        )
    ):

        cohort = df.filter(
            pl.col(from_col)
            .is_in(considered_buckets)
        )

        # ----------------------------------------------------
        # Cohort denominator
        # ----------------------------------------------------

        if account_col:

            cohort_sizes = (
                cohort
                .group_by(from_col)
                .agg(
                    pl.col(account_col)
                    .n_unique()
                    .alias(
                        "cohort_size"
                    )
                )
            )

        else:

            cohort_sizes = (
                cohort
                .group_by(from_col)
                .len(
                    name="cohort_size"
                )
            )

        # ----------------------------------------------------
        # Future snapshots
        # ----------------------------------------------------

        for to_col in target_cols[
            position + 1:
        ]:

            transitions = cohort.filter(
                pl.col(to_col)
                .is_in(
                    considered_buckets
                )
                |
                pl.col(to_col)
                .is_null()
            )

            if account_col:

                counts = (
                    transitions
                    .group_by(
                        [
                            from_col,
                            to_col,
                        ]
                    )
                    .agg(
                        pl.col(
                            account_col
                        )
                        .n_unique()
                        .alias(
                            "accounts"
                        )
                    )
                )

            else:

                counts = (
                    transitions
                    .group_by(
                        [
                            from_col,
                            to_col,
                        ]
                    )
                    .len(
                        name="accounts"
                    )
                )

            transition_result = (
                counts
                .join(
                    cohort_sizes,
                    on=from_col,
                    how="left",
                )
                .rename({
                    from_col: "from_bucket",
                    to_col: "to_bucket",
                })
                .with_columns(
                    pl.lit(from_col)
                    .alias(
                        "from_snapshot"
                    ),

                    pl.lit(to_col)
                    .alias(
                        "to_snapshot"
                    ),

                    (
                        pl.col("accounts")
                        /
                        pl.col("cohort_size")
                    )
                    .alias(
                        "roll_rate"
                    ),
                )
                .select([
                    "from_snapshot",
                    "from_bucket",
                    "to_snapshot",
                    "to_bucket",
                    "accounts",
                    "cohort_size",
                    "roll_rate",
                ])
            )

            results.append(
                transition_result
            )

    if not results:

        return pl.DataFrame()

    return pl.concat(
        results,
        how="vertical_relaxed",
    )


# ============================================================
# ROLL RATE MATRIX
# ============================================================

def cohort_matrix(
    roll_rates: pl.DataFrame,
    target_cols: list[str],
    from_snapshot: str,
    from_bucket: int,
    value: str = "accounts",
) -> pl.DataFrame:

    future_snapshots = target_cols[
        target_cols.index(
            from_snapshot
        ) + 1:
    ]

    cohort = roll_rates.filter(
        (
            pl.col("from_snapshot")
            == from_snapshot
        )
        &
        (
            pl.col("from_bucket")
            == from_bucket
        )
    )

    aggregations = [
        pl.col(value)
        .filter(
            pl.col("to_bucket")
            == bucket
        )
        .sum()
        .alias(str(bucket))
        for bucket in range(5)
    ]

    # Closed
    aggregations.append(
        pl.col(value)
        .filter(
            pl.col("to_bucket")
            .is_null()
        )
        .sum()
        .alias("closed")
    )

    matrix = (
        cohort
        .group_by("to_snapshot")
        .agg(aggregations)
    )

    snapshot_order = pl.DataFrame({
        "to_snapshot":
            future_snapshots,

        "_order":
            range(
                len(
                    future_snapshots
                )
            ),
    })

    output_columns = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "closed",
    ]

    return (
        snapshot_order
        .join(
            matrix,
            on="to_snapshot",
            how="left",
        )
        .sort("_order")
        .drop("_order")
        .with_columns(
            pl.col(
                output_columns
            )
            .fill_null(0)
        )
    )


# ============================================================
# ROLL RATE HIGHLIGHT TEST
# ============================================================

def roll_forward_exceeds_roll_back(
    row: dict,
) -> bool:

    # --------------------------------------------------------
    # Stabilization / roll-forward:
    #
    # immediate next bucket + same bucket
    #
    # compared against:
    #
    # roll-back buckets
    #
    # Closed accounts are deliberately excluded.
    # --------------------------------------------------------

    values = [
        float(
            row.get(str(i), 0) or 0
        )
        for i in range(5)
    ]

    # Need a from-bucket indicator
    from_bucket = row.get(
        "from_bucket"
    )

    if from_bucket is None:
        return False

    try:
        from_bucket = int(
            from_bucket
        )
    except:
        return False

    if from_bucket >= 4:
        return False

    immediate_next = (
        values[from_bucket + 1]
        if from_bucket + 1 <= 4
        else 0
    )

    roll_back = sum(
        values[
            :from_bucket
        ]
    )

    return (
        immediate_next
        <
        roll_back
    )


# ============================================================
# CAPTURE / CONVERSION
# ============================================================

def calculate_capture_conversion(
    df: pl.DataFrame,
    max_cols: list[str],
    account_col: str | None = None,
) -> pl.DataFrame:

    thresholds = [
        1,
        2,
        3,
        4,
    ]

    target_col = max_cols[-1]

    def count_population(
        condition: pl.Expr,
    ) -> int:

        filtered = df.filter(
            condition
        )

        if account_col:

            return int(
                filtered
                .filter(
                    pl.col(
                        account_col
                    ).is_not_null()
                )
                .select(
                    pl.col(
                        account_col
                    )
                    .n_unique()
                )
                .item()
            )

        return filtered.height

    target_condition = (
        pl.col(target_col)
        >= 4
    )

    total_target_accounts = (
        count_population(
            target_condition
        )
    )

    results = []

    for source_col in max_cols:

        for threshold in thresholds:

            source_condition = (
                pl.col(source_col)
                >= threshold
            )

            captured_condition = (
                source_condition
                &
                target_condition
            )

            source_accounts = (
                count_population(
                    source_condition
                )
            )

            captured_accounts = (
                count_population(
                    captured_condition
                )
            )

            capture_rate = (
                captured_accounts
                /
                total_target_accounts
                if total_target_accounts > 0
                else None
            )

            conversion_rate = (
                captured_accounts
                /
                source_accounts
                if source_accounts > 0
                else None
            )

            results.append({
                "IRAC Column":
                    source_col,

                "Threshold":
                    threshold,

                "Cut-off":
                    f"{threshold}+",

                "Captured Accounts":
                    captured_accounts,

                "Target 4+ Accounts":
                    total_target_accounts,

                "Source Accounts":
                    source_accounts,

                "Capture Rate":
                    capture_rate,

                "Conversion Rate":
                    conversion_rate,
            })

    return pl.DataFrame(
        results
    )


# ============================================================
# WATERFALL DATA
# ============================================================

def calculate_waterfall_counts(
    df: pl.DataFrame,
    max_cols: list[str],
    openingdt_col: str,
    end_dates: list[date | None],
    account_col: str | None = None,
) -> pl.DataFrame:

    results = []

    def count_rows(
        data: pl.DataFrame,
    ) -> int:

        if account_col:

            return int(
                data
                .filter(
                    pl.col(
                        account_col
                    ).is_not_null()
                )
                .select(
                    pl.col(
                        account_col
                    )
                    .n_unique()
                )
                .item()
            )

        return data.height

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    total = count_rows(df)

    results.append({
        "Period": "Total",
        "Count": total,
    })

    # --------------------------------------------------------
    # Datewise
    # --------------------------------------------------------

    for end_date in end_dates:

        if end_date is None:
            continue

        period_df = filter_by_openingdt(
            df,
            openingdt_col,
            end_date,
        )

        results.append({
            "Period":
                end_date.strftime(
                    "%Y-%m-%d"
                ),

            "Count":
                count_rows(
                    period_df
                ),
        })

    return pl.DataFrame(
        results
    )


# ============================================================
# EXCEL FORMATS
# ============================================================

def create_excel_formats(
    workbook,
) -> dict:

    return {

        "title":
            workbook.add_format({
                "bold": True,
                "font_size": 14,
                "font_color": "#FFFFFF",
                "bg_color": "#1F4E78",
                "align": "center",
                "valign": "vcenter",
                "border": 1,
            }),

        "subtitle":
            workbook.add_format({
                "italic": True,
                "font_color": "#595959",
            }),

        "header":
            workbook.add_format({
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": "#4472C4",
                "align": "center",
                "valign": "vcenter",
                "border": 1,
            }),

        "subheader":
            workbook.add_format({
                "bold": True,
                "bg_color": "#D9EAF7",
                "align": "center",
                "valign": "vcenter",
                "border": 1,
            }),

        "cutoff":
            workbook.add_format({
                "bold": True,
                "bg_color": "#F2F2F2",
                "align": "center",
                "border": 1,
            }),

        "rate":
            workbook.add_format({
                "num_format": "0.0%",
                "align": "center",
                "border": 1,
            }),

        "count":
            workbook.add_format({
                "num_format": "#,##0",
                "align": "center",
                "border": 1,
            }),

        "group":
            workbook.add_format({
                "bold": True,
                "font_size": 12,
                "font_color": "#FFFFFF",
                "bg_color": "#8064A2",
                "align": "left",
                "valign": "vcenter",
                "border": 1,
            }),

        "highlight":
            workbook.add_format({
                "bg_color": "#FFF2CC",
                "bold": True,
                "border": 1,
            }),

        "normal":
            workbook.add_format({
                "border": 1,
                "align": "center",
            }),

        "waterfall":
            workbook.add_format({
                "num_format": "#,##0",
                "align": "center",
                "border": 1,
            }),
    }


# ============================================================
# WRITE MAX IRAC SECTION
# ============================================================

def write_max_irac_section(
    worksheet,
    start_row: int,
    count_df: pl.DataFrame,
    rate_df: pl.DataFrame,
    max_cols: list[str],
    period_text: str,
    formats: dict,
) -> int:

    row = start_row

    # --------------------------------------------------------
    # Section title
    # --------------------------------------------------------

    last_col = max(
        len(max_cols),
        len(max_cols) * 2,
    )

    worksheet.merge_range(
        row,
        0,
        row,
        last_col,
        f"Max IRAC Analysis | {period_text}",
        formats["title"],
    )

    row += 2

    # ========================================================
    # COUNT TABLE
    # ========================================================

    worksheet.write(
        row,
        0,
        "IRAC Status",
        formats["header"],
    )

    for i, col in enumerate(max_cols):

        worksheet.write(
            row,
            i + 1,
            col,
            formats["header"],
        )

    row += 1

    for data_row in count_df.iter_rows(
        named=True
    ):

        value = data_row["value"]

        worksheet.write(
            row,
            0,
            value,
            formats["cutoff"],
        )

        for i, col in enumerate(max_cols):

            count = data_row[
                f"{col}_count"
            ]

            worksheet.write_number(
                row,
                i + 1,
                int(count),
                formats["count"],
            )

        row += 1

    row += 2

    # ========================================================
    # RATE TABLE
    # Rows = X+, 30+, 60+, 90+
    # Columns = Vintage
    # ========================================================

    rate_start_row = row

    worksheet.merge_range(
        row,
        0,
        row,
        len(max_cols),
        "Vintage Rate Table",
        formats["title"],
    )

    row += 1

    worksheet.write(
        row,
        0,
        "Cut-off",
        formats["header"],
    )

    for i, col in enumerate(max_cols):

        worksheet.write(
            row,
            i + 1,
            col,
            formats["header"],
        )

    row += 1

    rate_first_data_row = row

    for rate_row in rate_df.iter_rows(
        named=True
    ):

        worksheet.write(
            row,
            0,
            rate_row["Cut-off"],
            formats["cutoff"],
        )

        for i, col in enumerate(max_cols):

            value = rate_row[col]

            if value is None:

                worksheet.write_blank(
                    row,
                    i + 1,
                    None,
                    formats["rate"],
                )

            else:

                worksheet.write_number(
                    row,
                    i + 1,
                    float(value),
                    formats["rate"],
                )

        row += 1

    # ========================================================
    # VINTAGE CHART
    # ========================================================

    chart = worksheet.book.add_chart({
        "type": "line"
    })

    for i, rate_row in enumerate(
        range(4)
    ):

        chart.add_series({
            "name": [
                worksheet.name,
                rate_first_data_row
                + rate_row,
                0,
            ],

            "categories": [
                worksheet.name,
                rate_start_row + 1,
                1,
                rate_start_row + 1,
                len(max_cols),
            ],

            "values": [
                worksheet.name,
                rate_first_data_row
                + rate_row,
                1,
                rate_first_data_row
                + rate_row,
                len(max_cols),
            ],
        })

    chart.set_title({
        "name":
            f"Vintage Curves | {period_text}"
    })

    chart.set_x_axis({
        "name": "Vintage"
    })

    chart.set_y_axis({
        "name": "Rate",
        "num_format": "0.0%",
    })

    chart.set_legend({
        "position": "bottom"
    })

    chart.set_size({
        "width": 720,
        "height": 360,
    })

    chart_col = len(max_cols) + 3

    worksheet.insert_chart(
        rate_start_row,
        chart_col,
        chart,
    )

    return row + 2


# ============================================================
# WRITE MAX IRAC SHEET
# ============================================================

def write_max_irac_sheet(
    df: pl.DataFrame,
    max_cols: list[str],
    writer,
    sheet_name: str,
    config: IRACConfig,
    segment_value=None,
) -> None:

    workbook = writer.book

    formats = create_excel_formats(
        workbook
    )

    worksheet = workbook.add_worksheet(
        sheet_name
    )

    writer.sheets[
        sheet_name
    ] = worksheet

    # --------------------------------------------------------
    # Prepare only necessary columns
    # --------------------------------------------------------

    analysis_df = prepare_analysis_df(
        df=df,
        max_cols=max_cols,
        segment_col=config.segment_col,
        openingdt_col=config.openingdt_col,
        account_col=None,
    )

    analysis_df = prepare_openingdt(
        analysis_df,
        config.openingdt_col,
    )

    # --------------------------------------------------------
    # Segment filter
    # --------------------------------------------------------

    analysis_df = filter_segment(
        analysis_df,
        config.segment_col,
        segment_value,
    )

    # --------------------------------------------------------
    # Date filters
    # --------------------------------------------------------

    end_dates = normalize_end_dates(
        config.end_dates
    )

    current_row = 0

    # --------------------------------------------------------
    # Sheet heading
    # --------------------------------------------------------

    display_segment = (
        "Overall"
        if config.segment_col is None
        or segment_value is None
        else str(segment_value)
    )

    worksheet.merge_range(
        current_row,
        0,
        current_row,
        len(max_cols) + 8,
        (
            f"MAX IRAC ANALYSIS | "
            f"{display_segment}"
        ),
        formats["title"],
    )

    current_row += 2

    # --------------------------------------------------------
    # Process date periods
    # --------------------------------------------------------

    for end_date in tqdm(
        end_dates,
        desc=f"Max IRAC | {sheet_name}",
        unit="period",
    ):

        period_df = filter_by_openingdt(
            analysis_df,
            config.openingdt_col,
            end_date,
        )

        period_text = (
            "Entire Dataset"
            if end_date is None
            else
            f"{config.openingdt_col} <= "
            f"{end_date:%Y-%m-%d}"
        )

        print(
            f"{sheet_name} | "
            f"{period_text} | "
            f"Rows={period_df.height:,}"
        )

        count_df, rate_df = (
            calculate_max_irac_analysis(
                period_df,
                max_cols,
            )
        )

        current_row = (
            write_max_irac_section(
                worksheet=worksheet,
                start_row=current_row,
                count_df=count_df,
                rate_df=rate_df,
                max_cols=max_cols,
                period_text=period_text,
                formats=formats,
            )
        )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    worksheet.freeze_panes(
        3,
        1,
    )

    worksheet.set_column(
        0,
        len(max_cols) + 10,
        14,
    )


# ============================================================
# WRITE ROLL RATE SHEET
# ============================================================

def write_roll_rate_sheet(
    df: pl.DataFrame,
    target_cols: list[str],
    writer,
    sheet_name: str,
    config: IRACConfig,
    segment_value=None,
) -> None:

    workbook = writer.book

    formats = create_excel_formats(
        workbook
    )

    worksheet = workbook.add_worksheet(
        sheet_name
    )

    writer.sheets[
        sheet_name
    ] = worksheet

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    analysis_df = prepare_analysis_df(
        df=df,
        max_cols=target_cols,
        segment_col=config.segment_col,
        openingdt_col=config.openingdt_col,
        account_col=config.account_col,
    )

    analysis_df = prepare_openingdt(
        analysis_df,
        config.openingdt_col,
    )

    analysis_df = filter_segment(
        analysis_df,
        config.segment_col,
        segment_value,
    )

    end_dates = normalize_end_dates(
        config.end_dates
    )

    current_row = 0

    display_segment = (
        "Overall"
        if config.segment_col is None
        or segment_value is None
        else str(segment_value)
    )

    worksheet.merge_range(
        current_row,
        0,
        current_row,
        10,
        (
            f"ROLL RATE ANALYSIS | "
            f"{display_segment}"
        ),
        formats["title"],
    )

    current_row += 2

    # --------------------------------------------------------
    # Each opening date period
    # --------------------------------------------------------

    for period_index, end_date in enumerate(
        tqdm(
            end_dates,
            desc=f"Roll Rate | {sheet_name}",
            unit="period",
        )
    ):

        period_df = filter_by_openingdt(
            analysis_df,
            config.openingdt_col,
            end_date,
        )

        period_text = (
            "Entire Dataset"
            if end_date is None
            else
            f"{config.openingdt_col} <= "
            f"{end_date:%Y-%m-%d}"
        )

        print(
            f"Roll rate | "
            f"{sheet_name} | "
            f"{period_text} | "
            f"Rows={period_df.height:,}"
        )

        roll_df = generate_roll_rates(
            period_df,
            target_cols,
            account_col=config.account_col,
            considered_buckets=config.roll_buckets,
        )

        if roll_df.height == 0:

            continue

        # ----------------------------------------------------
        # Section heading
        # ----------------------------------------------------

        worksheet.merge_range(
            current_row,
            0,
            current_row,
            9,
            period_text,
            formats["title"],
        )

        current_row += 2

        # ----------------------------------------------------
        # Buckets 0-4
        # ----------------------------------------------------

        for from_snapshot in target_cols[:-1]:

            for from_bucket in (
                config.roll_from_buckets
            ):

                count_matrix = cohort_matrix(
                    roll_df,
                    target_cols,
                    from_snapshot,
                    from_bucket,
                    "accounts",
                )

                rate_matrix = cohort_matrix(
                    roll_df,
                    target_cols,
                    from_snapshot,
                    from_bucket,
                    "roll_rate",
                ).with_columns(
                    pl.col([
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "closed",
                    ])
                    .round(4)
                )

                # ------------------------------------------------
                # Merge count and rate
                # ------------------------------------------------

                count_pd = (
                    count_matrix
                    .to_pandas()
                )

                rate_pd = (
                    rate_matrix
                    .to_pandas()
                )

                # ------------------------------------------------
                # Write heading
                # ------------------------------------------------

                worksheet.write(
                    current_row,
                    0,
                    (
                        f"{from_snapshot} | "
                        f"Bucket {from_bucket}"
                    ),
                    formats["group"],
                )

                current_row += 1

                # ------------------------------------------------
                # Count table
                # ------------------------------------------------

                count_pd.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    startrow=current_row,
                    startcol=0,
                    index=False,
                )

                current_row += (
                    len(count_pd)
                    + 3
                )

                # ------------------------------------------------
                # Rate table
                # ------------------------------------------------

                rate_start_row = (
                    current_row
                )

                rate_pd.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    startrow=rate_start_row,
                    startcol=0,
                    index=False,
                )

                # ------------------------------------------------
                # Highlight condition
                #
                # We deliberately do not include CLOSED.
                # ------------------------------------------------

                for excel_offset, rate_row in enumerate(
                    rate_matrix.iter_rows(
                        named=True
                    ),
                    start=1,
                ):

                    values = [
                        float(
                            rate_row.get(
                                str(i),
                                0
                            )
                            or 0
                        )
                        for i in range(5)
                    ]

                    immediate_next = (
                        values[
                            from_bucket + 1
                        ]
                        if from_bucket < 4
                        else 0
                    )

                    roll_back = sum(
                        values[:from_bucket]
                    )

                    if (
                        immediate_next
                        <
                        roll_back
                    ):

                        worksheet.set_row(
                            rate_start_row
                            + excel_offset,
                            None,
                            formats[
                                "highlight"
                            ],
                        )

                current_row += (
                    len(rate_pd)
                    + 2
                )

            current_row += (
                config.roll_rate_gap_columns
            )

    worksheet.freeze_panes(
        3,
        0,
    )

    worksheet.set_column(
        0,
        20,
        15,
    )


# ============================================================
# WRITE CAPTURE / CONVERSION SHEET
# ============================================================

def write_capture_conversion_sheet(
    df: pl.DataFrame,
    max_cols: list[str],
    writer,
    sheet_name: str,
    config: IRACConfig,
    segment_value=None,
) -> None:

    workbook = writer.book

    formats = create_excel_formats(
        workbook
    )

    worksheet = workbook.add_worksheet(
        sheet_name
    )

    writer.sheets[
        sheet_name
    ] = worksheet

    analysis_df = prepare_analysis_df(
        df=df,
        max_cols=max_cols,
        segment_col=config.segment_col,
        openingdt_col=config.openingdt_col,
        account_col=config.account_col,
    )

    analysis_df = prepare_openingdt(
        analysis_df,
        config.openingdt_col,
    )

    analysis_df = filter_segment(
        analysis_df,
        config.segment_col,
        segment_value,
    )

    end_dates = normalize_end_dates(
        config.end_dates
    )

    display_segment = (
        "Overall"
        if config.segment_col is None
        or segment_value is None
        else str(segment_value)
    )

    current_row = 0

    worksheet.merge_range(
        current_row,
        0,
        current_row,
        len(max_cols) * 2 + 2,
        (
            f"CAPTURE / CONVERSION | "
            f"{display_segment}"
        ),
        formats["title"],
    )

    current_row += 2

    # --------------------------------------------------------
    # Process each period
    # --------------------------------------------------------

    for end_date in tqdm(
        end_dates,
        desc=f"Capture Conversion | {sheet_name}",
        unit="period",
    ):

        period_df = filter_by_openingdt(
            analysis_df,
            config.openingdt_col,
            end_date,
        )

        period_text = (
            "Entire Dataset"
            if end_date is None
            else
            f"{config.openingdt_col} <= "
            f"{end_date:%Y-%m-%d}"
        )

        result_df = (
            calculate_capture_conversion(
                period_df,
                max_cols,
                config.account_col,
            )
        )

        result_df = result_df.with_columns(
            pl.lit(
                period_text
            ).alias(
                "Period"
            )
        )

        # ----------------------------------------------------
        # Heading
        # ----------------------------------------------------

        worksheet.merge_range(
            current_row,
            0,
            current_row,
            10,
            period_text,
            formats["title"],
        )

        current_row += 2

        # ----------------------------------------------------
        # Rate matrix
        # ----------------------------------------------------

        rate_lookup = {
            (
                row["IRAC Column"],
                row["Threshold"],
            ): row
            for row in result_df.iter_rows(
                named=True
            )
        }

        worksheet.write(
            current_row,
            0,
            "Cut-off",
            formats["header"],
        )

        for position, col in enumerate(
            max_cols
        ):

            capture_col = (
                1 + position * 2
            )

            conversion_col = (
                capture_col + 1
            )

            worksheet.merge_range(
                current_row,
                capture_col,
                current_row,
                conversion_col,
                col,
                formats["header"],
            )

            worksheet.write(
                current_row + 1,
                capture_col,
                "Capture",
                formats["subheader"],
            )

            worksheet.write(
                current_row + 1,
                conversion_col,
                "Conversion",
                formats["subheader"],
            )

        current_row += 2

        for threshold in [1, 2, 3, 4]:

            worksheet.write(
                current_row,
                0,
                f"{threshold}+",
                formats["cutoff"],
            )

            for position, col in enumerate(
                max_cols
            ):

                capture_col = (
                    1 + position * 2
                )

                conversion_col = (
                    capture_col + 1
                )

                result = rate_lookup[
                    (col, threshold)
                ]

                capture = result[
                    "Capture Rate"
                ]

                conversion = result[
                    "Conversion Rate"
                ]

                if capture is not None:

                    worksheet.write_number(
                        current_row,
                        capture_col,
                        float(capture),
                        formats["rate"],
                    )

                if conversion is not None:

                    worksheet.write_number(
                        current_row,
                        conversion_col,
                        float(conversion),
                        formats["rate"],
                    )

            current_row += 1

        # ----------------------------------------------------
        # Counts
        # ----------------------------------------------------

        current_row += 2

        worksheet.write(
            current_row,
            0,
            "Cut-off",
            formats["header"],
        )

        for position, col in enumerate(
            max_cols
        ):

            worksheet.write(
                current_row,
                1 + position * 2,
                f"{col} Captured 4+",
                formats["header"],
            )

            worksheet.write(
                current_row,
                2 + position * 2,
                f"{col} Source",
                formats["header"],
            )

        current_row += 1

        for threshold in [1, 2, 3, 4]:

            worksheet.write(
                current_row,
                0,
                f"{threshold}+",
                formats["cutoff"],
            )

            for position, col in enumerate(
                max_cols
            ):

                captured_col = (
                    1 + position * 2
                )

                source_col = (
                    captured_col + 1
                )

                result = rate_lookup[
                    (col, threshold)
                ]

                worksheet.write_number(
                    current_row,
                    captured_col,
                    int(
                        result[
                            "Captured Accounts"
                        ]
                    ),
                    formats["count"],
                )

                worksheet.write_number(
                    current_row,
                    source_col,
                    int(
                        result[
                            "Source Accounts"
                        ]
                    ),
                    formats["count"],
                )

            current_row += 1

        current_row += 3

    worksheet.freeze_panes(
        3,
        1,
    )

    worksheet.set_column(
        0,
        len(max_cols) * 2 + 3,
        16,
    )


# ============================================================
# WRITE WATERFALL SHEET
# ============================================================

def write_waterfall_sheet(
    df: pl.DataFrame,
    max_cols: list[str],
    writer,
    config: IRACConfig,
) -> None:

    workbook = writer.book

    formats = create_excel_formats(
        workbook
    )

    sheet_name = "Waterfall"

    worksheet = workbook.add_worksheet(
        sheet_name
    )

    writer.sheets[
        sheet_name
    ] = worksheet

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    analysis_df = prepare_analysis_df(
        df=df,
        max_cols=max_cols,
        segment_col=config.segment_col,
        openingdt_col=config.openingdt_col,
        account_col=config.account_col,
    )

    analysis_df = prepare_openingdt(
        analysis_df,
        config.openingdt_col,
    )

    end_dates = normalize_end_dates(
        config.end_dates
    )

    segments = get_segment_values(
        analysis_df,
        config.segment_col,
    )

    # Add overall
    groups = [
        ("Overall", None)
    ]

    if config.segment_col:

        for segment in segments:

            groups.append(
                (
                    str(segment)
                    if segment is not None
                    else "NULL",
                    segment,
                )
            )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    worksheet.write(
        0,
        0,
        "Period",
        formats["header"],
    )

    for col_idx, (
        display_name,
        segment_value,
    ) in enumerate(
        groups,
        start=1,
    ):

        worksheet.write(
            0,
            col_idx,
            display_name,
            formats["header"],
        )

    # --------------------------------------------------------
    # Rows
    # --------------------------------------------------------

    row = 1

    periods = [
        None
    ]

    periods.extend(
        [
            x
            for x in end_dates
            if x is not None
        ]
    )

    for end_date in periods:

        period_label = (
            "Total"
            if end_date is None
            else end_date.strftime(
                "%Y-%m-%d"
            )
        )

        worksheet.write(
            row,
            0,
            period_label,
            formats["cutoff"],
        )

        for col_idx, (
            display_name,
            segment_value,
        ) in enumerate(
            groups,
            start=1,
        ):

            period_df = (
                analysis_df
                if end_date is None
                else filter_by_openingdt(
                    analysis_df,
                    config.openingdt_col,
                    end_date,
                )
            )

            if segment_value is not None:

                period_df = filter_segment(
                    period_df,
                    config.segment_col,
                    segment_value,
                )

            elif (
                config.segment_col
                and display_name == "NULL"
            ):

                period_df = filter_segment(
                    period_df,
                    config.segment_col,
                    None,
                )

            if config.account_col:

                count = int(
                    period_df
                    .filter(
                        pl.col(
                            config.account_col
                        )
                        .is_not_null()
                    )
                    .select(
                        pl.col(
                            config.account_col
                        )
                        .n_unique()
                    )
                    .item()
                )

            else:

                count = period_df.height

            worksheet.write_number(
                row,
                col_idx,
                count,
                formats["waterfall"],
            )

        row += 1

    worksheet.freeze_panes(
        1,
        1,
    )

    worksheet.set_column(
        0,
        len(groups),
        18,
    )


# ============================================================
# BUILD COMPLETE EXCEL REPORT
# ============================================================

def build_excel_report(
    df: pl.DataFrame,
    max_cols: list[str],
    config: IRACConfig,
) -> None:

    with Timer("BUILD COMPLETE EXCEL REPORT"):

        output_path = Path(
            config.output_file
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Prepare lightweight dataframe once
        # ----------------------------------------------------

        required_cols = list(
            dict.fromkeys(
                [
                    *max_cols,
                    config.segment_col,
                    config.openingdt_col,
                    config.account_col,
                ]
            )
        )

        required_cols = [
            col
            for col in required_cols
            if col is not None
        ]

        missing = [
            col
            for col in required_cols
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing required columns: {missing}"
            )

        report_df = df.select(
            required_cols
        )

        report_df = prepare_openingdt(
            report_df,
            config.openingdt_col,
        )

        # ----------------------------------------------------
        # Segment values
        # ----------------------------------------------------

        if config.segment_col:

            segments = get_segment_values(
                report_df,
                config.segment_col,
            )

        else:

            segments = []

        # ----------------------------------------------------
        # Excel
        # ----------------------------------------------------

        with pd.ExcelWriter(
            output_path,
            engine="xlsxwriter",
        ) as writer:

            # =================================================
            # MAX IRAC
            # =================================================

            print(
                "\n"
                + "=" * 70
            )

            print(
                "BUILDING MAX IRAC SHEETS"
            )

            print(
                "=" * 70
            )

            # Overall
            write_max_irac_sheet(
                df=report_df,
                max_cols=max_cols,
                writer=writer,
                sheet_name="Overall - Max IRAC",
                config=config,
                segment_value=None,
            )

            # Product sheets
            for segment in tqdm(
                segments,
                desc="Max IRAC product sheets",
                unit="sheet",
            ):

                display_name = (
                    "NULL"
                    if segment is None
                    else str(segment)
                )

                sheet_name = (
                    f"Max - {display_name}"
                )[:31]

                write_max_irac_sheet(
                    df=report_df,
                    max_cols=max_cols,
                    writer=writer,
                    sheet_name=sheet_name,
                    config=config,
                    segment_value=segment,
                )

            # =================================================
            # ROLL RATES
            # =================================================

            print(
                "\n"
                + "=" * 70
            )

            print(
                "BUILDING ROLL RATE SHEETS"
            )

            print(
                "=" * 70
            )

            write_roll_rate_sheet(
                df=report_df,
                target_cols=max_cols,
                writer=writer,
                sheet_name="Overall - Roll Rates",
                config=config,
                segment_value=None,
            )

            for segment in tqdm(
                segments,
                desc="Roll rate product sheets",
                unit="sheet",
            ):

                display_name = (
                    "NULL"
                    if segment is None
                    else str(segment)
                )

                sheet_name = (
                    f"Roll - {display_name}"
                )[:31]

                write_roll_rate_sheet(
                    df=report_df,
                    target_cols=max_cols,
                    writer=writer,
                    sheet_name=sheet_name,
                    config=config,
                    segment_value=segment,
                )

            # =================================================
            # CAPTURE / CONVERSION
            # =================================================

            print(
                "\n"
                + "=" * 70
            )

            print(
                "BUILDING CAPTURE / CONVERSION SHEETS"
            )

            print(
                "=" * 70
            )

            write_capture_conversion_sheet(
                df=report_df,
                max_cols=max_cols,
                writer=writer,
                sheet_name="Overall - Capture",
                config=config,
                segment_value=None,
            )

            for segment in tqdm(
                segments,
                desc="Capture product sheets",
                unit="sheet",
            ):

                display_name = (
                    "NULL"
                    if segment is None
                    else str(segment)
                )

                sheet_name = (
                    f"Capture - {display_name}"
                )[:31]

                write_capture_conversion_sheet(
                    df=report_df,
                    max_cols=max_cols,
                    writer=writer,
                    sheet_name=sheet_name,
                    config=config,
                    segment_value=segment,
                )

            # =================================================
            # WATERFALL
            # =================================================

            print(
                "\n"
                + "=" * 70
            )

            print(
                "BUILDING WATERFALL"
            )

            print(
                "=" * 70
            )

            write_waterfall_sheet(
                df=report_df,
                max_cols=max_cols,
                writer=writer,
                config=config,
            )

        print(
            f"\nExcel report saved to:\n"
            f"{output_path}"
        )


# ============================================================
# MASTER FUNCTION
# ============================================================

def run_irac_analysis(
    config: IRACConfig,
) -> None:

    total_start = time.perf_counter()

    print(
        "\n"
        + "#" * 80
    )

    print(
        "# IRAC ANALYSIS PIPELINE"
    )

    print(
        "#" * 80
    )

    # --------------------------------------------------------
    # 1. Read
    # --------------------------------------------------------

    df = read_input_data(
        input_file=config.input_file,
        separator=config.separator,
    )

    # --------------------------------------------------------
    # 2. Create max IRAC flags
    # --------------------------------------------------------

    df, max_cols = (
        create_max_irac_flags(
            df=df,
            irac_suffix=config.irac_suffix,
            max_columns=config.max_irac_columns,
            drop_first=True,
        )
    )

    # --------------------------------------------------------
    # 3. Build Excel
    # --------------------------------------------------------

    build_excel_report(
        df=df,
        max_cols=max_cols,
        config=config,
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - total_start
    )

    print(
        "\n"
        + "#" * 80
    )

    print(
        "IRAC ANALYSIS COMPLETE"
    )

    print(
        f"Total execution time: "
        f"{elapsed:,.2f} seconds"
    )

    print(
        f"Output file: "
        f"{config.output_file}"
    )

    print(
        "#" * 80
    )


# ============================================================
# RUN
# ============================================================

config = IRACConfig(

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    input_file=(
        r"D:/DELOITE_RISK_SCORE_CARD_DATA_V2.TXT"
    ),

    output_file=(
        r"C:/Users/VC2003776/Documents/"
        r"PLSE AS/analystics_data/"
        r"IRAC_Analysis_Report.xlsx"
    ),

    separator="|",

    # --------------------------------------------------------
    # Columns
    # --------------------------------------------------------

    account_col="LON_ACCT_NBR",

    openingdt_col="OPENINGDT",

    segment_col="PRODUCT_TYPE",

    # --------------------------------------------------------
    # IRAC
    # --------------------------------------------------------

    irac_suffix="_IRAC",

    max_irac_columns=10,

    # --------------------------------------------------------
    # Opening date filters
    # --------------------------------------------------------

    end_dates=[
        "2025-03-31",
        "2025-06-30",
        "2025-09-30",
        "2025-12-31",
    ],

    # --------------------------------------------------------
    # Roll rate
    # --------------------------------------------------------

    roll_buckets=(
        0,
        1,
        2,
        3,
        4,
    ),

    roll_from_buckets=(
        2,
        3,
    ),

    roll_rate_gap_columns=4,
)


# ============================================================
# EXECUTE
# ============================================================

run_irac_analysis(config)
