import pandas as pd
import polars as pl

from datetime import date, datetime


def create_capture_conversion_excel(
    df: pl.DataFrame,
    max_cols: list[str],
    output_file: str = "irac_capture_conversion.xlsx",
    account_col: str | None = None,
    end_dates: list[str | date | datetime] | None = None,
    split_by_col: str | None = None,
    date_col: str = "OPENINGDT",
) -> pl.DataFrame:
    """
    Calculate capture and conversion rates for every IRAC column
    against two anchor definitions in the final IRAC / 24MOB column:

        4+
        5+

    Source cut-offs evaluated:

        1+, 2+, 3+, 4+, 5+

    Definitions
    -----------
    Captured accounts:
        Accounts meeting the source cut-off AND the selected
        anchor cut-off in the target column.

    Capture rate:
        Captured accounts / all accounts meeting the anchor
        cut-off in the target column.

    Conversion rate:
        Captured accounts / all accounts meeting the source
        cut-off.

    Null IRAC values:
        Null IRAC values are naturally excluded from threshold
        populations because comparisons such as >= 4 or >= 5
        evaluate to false.

    Account counting:
        When account_col is supplied, unique non-null accounts
        are counted. Otherwise dataframe rows are counted.

    Date handling
    -------------
    date_col identifies the date column used for end-date filtering.

    If end_dates is supplied:
        date_col <= end_date

    If end_dates is None:
        the entire dataset is analysed.

    split_by_col is optional.

    If split_by_col is supplied:
        Each end-date sheet is split into groups.

    If split_by_col is not supplied:
        Each end_date still gets its own worksheet.

    Excel output
    ------------
    Each end-date worksheet contains, for every group:

        1. 4+ anchor percentage matrix
        2. 5+ anchor percentage matrix
        3. 4+ anchor value-count matrix
        4. 5+ anchor value-count matrix

    The same worksheet therefore contains both 4+ and 5+
    comparisons.
    """

    # =========================================================
    # Configuration
    # =========================================================

    source_thresholds = [1, 2, 3, 4, 5]
    anchor_thresholds = [4, 5]

    # =========================================================
    # Validation
    # =========================================================

    if not max_cols:
        raise ValueError("max_cols cannot be empty.")

    if len(max_cols) != len(set(max_cols)):
        raise ValueError(
            "max_cols contains duplicate column names."
        )

    target_col = max_cols[-1]

    required_cols = list(max_cols)

    if account_col:
        required_cols.append(account_col)

    if end_dates:
        required_cols.append(date_col)

    if split_by_col:
        required_cols.append(split_by_col)

    missing_cols = sorted(
        set(required_cols) - set(df.columns)
    )

    if missing_cols:
        raise ValueError(
            f"Columns not found in dataframe: {missing_cols}"
        )

    # =========================================================
    # Normalize end dates
    # =========================================================

    def normalize_date(
        value: str | date | datetime,
    ) -> date:
        if isinstance(value, datetime):
            return value.date()

        if isinstance(value, date):
            return value

        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError:
                raise ValueError(
                    f"Invalid end date {value!r}. "
                    "Expected YYYY-MM-DD."
                )

        raise TypeError(
            "end_dates must contain str, date, or datetime values."
        )

    if end_dates:
        normalized_end_dates = [
            normalize_date(x)
            for x in end_dates
        ]

        # Remove duplicates while preserving order.
        normalized_end_dates = list(
            dict.fromkeys(normalized_end_dates)
        )

    else:
        # None means entire dataset.
        normalized_end_dates = [None]

    # =========================================================
    # Prepare analysis dataframe
    # =========================================================

    analysis_df = df

    if end_dates:
        date_dtype = analysis_df.schema[date_col]

        if date_dtype == pl.Date:
            analysis_df = analysis_df.with_columns(
                pl.col(date_col).alias("_analysis_date")
            )

        elif date_dtype == pl.Datetime:
            analysis_df = analysis_df.with_columns(
                pl.col(date_col)
                .dt.date()
                .alias("_analysis_date")
            )

        else:
            analysis_df = analysis_df.with_columns(
                pl.col(date_col)
                .cast(pl.String)
                .str.strptime(
                    pl.Date,
                    strict=False,
                )
                .alias("_analysis_date")
            )

    # =========================================================
    # Cast IRAC values to integers
    # =========================================================

    analysis_df = analysis_df.with_columns([
        pl.col(column)
        .cast(pl.Int64, strict=False)
        .alias(column)
        for column in max_cols
    ])

    # =========================================================
    # Analysis function
    # =========================================================

    def calculate_analysis(
        analysis_data: pl.DataFrame,
        group_value=None,
    ) -> pl.DataFrame:
        """
        Run capture/conversion analysis on one filtered
        Polars DataFrame.

        Both 4+ and 5+ anchors are calculated.
        """

        def count_population(
            condition: pl.Expr,
        ) -> int:
            """
            Count either unique accounts or dataframe rows
            satisfying the supplied condition.
            """

            filtered = analysis_data.filter(condition)

            if account_col:
                return int(
                    filtered
                    .filter(
                        pl.col(account_col).is_not_null()
                    )
                    .select(
                        pl.col(account_col).n_unique()
                    )
                    .item()
                )

            return filtered.height

        results = []

        # -----------------------------------------------------
        # Calculate both anchor definitions
        # -----------------------------------------------------

        for anchor_threshold in anchor_thresholds:

            anchor_condition = (
                pl.col(target_col) >= anchor_threshold
            )

            total_anchor_accounts = count_population(
                anchor_condition
            )

            # -------------------------------------------------
            # Calculate capture/conversion for each source
            # column and source cutoff.
            # -------------------------------------------------

            for source_col in max_cols:

                for source_threshold in source_thresholds:

                    source_condition = (
                        pl.col(source_col)
                        >= source_threshold
                    )

                    captured_condition = (
                        source_condition
                        & anchor_condition
                    )

                    source_accounts = count_population(
                        source_condition
                    )

                    captured_accounts = count_population(
                        captured_condition
                    )

                    capture_rate = (
                        captured_accounts
                        / total_anchor_accounts
                        if total_anchor_accounts > 0
                        else None
                    )

                    conversion_rate = (
                        captured_accounts
                        / source_accounts
                        if source_accounts > 0
                        else None
                    )

                    results.append({
                        "Group": group_value,
                        "IRAC Column": source_col,

                        "Source Threshold":
                            source_threshold,

                        "Source Cut-off":
                            f"{source_threshold}+",

                        "Anchor Threshold":
                            anchor_threshold,

                        "Anchor Cut-off":
                            f"{anchor_threshold}+",

                        "Captured Accounts":
                            captured_accounts,

                        "Target Accounts":
                            total_anchor_accounts,

                        "Source Accounts":
                            source_accounts,

                        "Capture Rate":
                            capture_rate,

                        "Conversion Rate":
                            conversion_rate,
                    })

        return pl.DataFrame(results)

    # =========================================================
    # Excel writer
    # =========================================================

    with pd.ExcelWriter(
        output_file,
        engine="xlsxwriter",
    ) as writer:

        workbook = writer.book

        # =====================================================
        # Formats
        # =====================================================

        title_format = workbook.add_format({
            "bold": True,
            "font_size": 14,
            "font_color": "#FFFFFF",
            "bg_color": "#1F4E78",
            "align": "center",
            "valign": "vcenter",
            "border": 1,
        })

        subtitle_format = workbook.add_format({
            "italic": True,
            "font_color": "#595959",
        })

        period_format = workbook.add_format({
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#4472C4",
            "align": "center",
            "valign": "vcenter",
            "border": 1,
        })

        subheader_format = workbook.add_format({
            "bold": True,
            "bg_color": "#D9EAF7",
            "align": "center",
            "valign": "vcenter",
            "border": 1,
        })

        cutoff_format = workbook.add_format({
            "bold": True,
            "bg_color": "#F2F2F2",
            "align": "center",
            "valign": "vcenter",
            "border": 1,
        })

        capture_format = workbook.add_format({
            "num_format": "0.0%",
            "bg_color": "#D9EAF7",
            "align": "center",
            "border": 1,
        })

        conversion_format = workbook.add_format({
            "num_format": "0.0%",
            "bg_color": "#E2F0D9",
            "align": "center",
            "border": 1,
        })

        count_title_format = workbook.add_format({
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#548235",
            "align": "center",
            "valign": "vcenter",
            "border": 1,
        })

        count_header_format = workbook.add_format({
            "bold": True,
            "bg_color": "#E2F0D9",
            "align": "center",
            "valign": "vcenter",
            "border": 1,
        })

        count_format = workbook.add_format({
            "num_format": "#,##0",
            "align": "center",
            "border": 1,
        })

        group_format = workbook.add_format({
            "bold": True,
            "font_size": 12,
            "font_color": "#FFFFFF",
            "bg_color": "#8064A2",
            "align": "left",
            "valign": "vcenter",
            "border": 1,
        })

        anchor_4_format = workbook.add_format({
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#4472C4",
            "align": "center",
            "valign": "vcenter",
            "border": 1,
        })

        anchor_5_format = workbook.add_format({
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#C55A11",
            "align": "center",
            "valign": "vcenter",
            "border": 1,
        })

        # =====================================================
        # Excel sheet name helper
        # =====================================================

        def make_sheet_name(
            end_date: date | None,
        ) -> str:

            if end_date is None:
                return "Capture Conversion"

            return (
                f"End {end_date.strftime('%Y-%m-%d')}"
            )

        # =====================================================
        # Combined return results
        # =====================================================

        all_results = []

        # =====================================================
        # One worksheet per end_date
        #
        # This works whether split_by_col is supplied or not.
        # =====================================================

        for end_date in normalized_end_dates:

            # -------------------------------------------------
            # Date filter
            # -------------------------------------------------

            if end_date is None:

                period_df = analysis_df

                period_text = "Entire Dataset"

            else:

                period_df = analysis_df.filter(
                    pl.col("_analysis_date")
                    <= pl.lit(end_date)
                )

                period_text = (
                    f"{date_col} <= "
                    f"{end_date.strftime('%Y-%m-%d')}"
                )

            # -------------------------------------------------
            # Determine groups
            # -------------------------------------------------

            if split_by_col:

                unique_groups = (
                    period_df
                    .select(split_by_col)
                    .unique(
                        maintain_order=True
                    )
                    .to_series()
                    .to_list()
                )

            else:

                # No split column:
                # still process this end_date once.
                unique_groups = [None]

            # -------------------------------------------------
            # Create worksheet
            # -------------------------------------------------

            sheet_name = make_sheet_name(end_date)

            worksheet = workbook.add_worksheet(
                sheet_name
            )

            writer.sheets[sheet_name] = worksheet

            last_excel_column = (
                len(max_cols) * 2
            )

            current_row = 0

            # =================================================
            # Sheet title
            # =================================================

            worksheet.merge_range(
                current_row,
                0,
                current_row,
                last_excel_column,
                (
                    "IRAC / 24MOB Capture and "
                    "Conversion Analysis"
                ),
                title_format,
            )

            current_row += 1

            subtitle_parts = [
                period_text,
                f"Target: {target_col}",
                "Anchors: 4+ and 5+",
            ]

            if split_by_col:
                subtitle_parts.append(
                    f"Split by: {split_by_col}"
                )

            worksheet.merge_range(
                current_row,
                0,
                current_row,
                last_excel_column,
                " | ".join(subtitle_parts),
                subtitle_format,
            )

            current_row += 2

            # =================================================
            # Process each group
            # =================================================

            for group_value in unique_groups:

                # -------------------------------------------------
                # Filter group
                # -------------------------------------------------

                if split_by_col:

                    if group_value is None:

                        group_df = period_df.filter(
                            pl.col(split_by_col)
                            .is_null()
                        )

                        display_group = "NULL"

                    else:

                        group_df = period_df.filter(
                            pl.col(split_by_col)
                            == group_value
                        )

                        display_group = str(
                            group_value
                        )

                else:

                    group_df = period_df
                    display_group = None

                # -------------------------------------------------
                # Run analysis
                # -------------------------------------------------

                result_df = calculate_analysis(
                    group_df,
                    group_value=group_value,
                )

                # Add end date to returned result.
                result_df = result_df.with_columns(
                    pl.lit(end_date).alias(
                        "End Date"
                    )
                )

                all_results.append(result_df)

                # -------------------------------------------------
                # Lookup for Excel rendering
                # -------------------------------------------------

                result_lookup = {
                    (
                        row["IRAC Column"],
                        row["Source Threshold"],
                        row["Anchor Threshold"],
                    ): row
                    for row in result_df.iter_rows(
                        named=True
                    )
                }

                # -------------------------------------------------
                # Get total anchor populations
                # -------------------------------------------------

                total_4_target_accounts = (
                    int(
                        result_df
                        .filter(
                            pl.col(
                                "Anchor Threshold"
                            ) == 4
                        )
                        .select(
                            pl.col(
                                "Target Accounts"
                            ).first()
                        )
                        .item()
                    )
                    if result_df.height > 0
                    and result_df.filter(
                        pl.col(
                            "Anchor Threshold"
                        ) == 4
                    ).height > 0
                    else 0
                )

                total_5_target_accounts = (
                    int(
                        result_df
                        .filter(
                            pl.col(
                                "Anchor Threshold"
                            ) == 5
                        )
                        .select(
                            pl.col(
                                "Target Accounts"
                            ).first()
                        )
                        .item()
                    )
                    if result_df.height > 0
                    and result_df.filter(
                        pl.col(
                            "Anchor Threshold"
                        ) == 5
                    ).height > 0
                    else 0
                )

                # =================================================
                # Group heading
                # =================================================

                if split_by_col:

                    worksheet.merge_range(
                        current_row,
                        0,
                        current_row,
                        last_excel_column,
                        (
                            f"{split_by_col}: "
                            f"{display_group}"
                        ),
                        group_format,
                    )

                    current_row += 2

                # =================================================
                # 4+ Anchor Percentage Matrix
                # =================================================

                worksheet.merge_range(
                    current_row,
                    0,
                    current_row,
                    last_excel_column,
                    (
                        "IRAC / 24MOB Capture and "
                        f"Conversion Matrix | "
                        f"Anchor: {target_col} 4+"
                    ),
                    anchor_4_format,
                )

                current_row += 2

                percentage_period_header_row = (
                    current_row
                )

                percentage_subheader_row = (
                    current_row + 1
                )

                percentage_first_data_row = (
                    current_row + 2
                )

                # -------------------------------------------------
                # Cut-off header
                # -------------------------------------------------

                worksheet.merge_range(
                    percentage_period_header_row,
                    0,
                    percentage_subheader_row,
                    0,
                    "Cut-off",
                    period_format,
                )

                # -------------------------------------------------
                # IRAC headers
                # -------------------------------------------------

                for position, source_col in enumerate(
                    max_cols
                ):

                    capture_column = (
                        1 + position * 2
                    )

                    conversion_column = (
                        capture_column + 1
                    )

                    worksheet.merge_range(
                        percentage_period_header_row,
                        capture_column,
                        percentage_period_header_row,
                        conversion_column,
                        source_col,
                        period_format,
                    )

                    worksheet.write(
                        percentage_subheader_row,
                        capture_column,
                        "Capture",
                        subheader_format,
                    )

                    worksheet.write(
                        percentage_subheader_row,
                        conversion_column,
                        "Conversion",
                        subheader_format,
                    )

                # -------------------------------------------------
                # 4+ percentage values
                # -------------------------------------------------

                for row_position, threshold in enumerate(
                    source_thresholds
                ):

                    excel_row = (
                        percentage_first_data_row
                        + row_position
                    )

                    worksheet.write(
                        excel_row,
                        0,
                        f"{threshold}+",
                        cutoff_format,
                    )

                    for position, source_col in enumerate(
                        max_cols
                    ):

                        capture_column = (
                            1 + position * 2
                        )

                        conversion_column = (
                            capture_column + 1
                        )

                        result = result_lookup[
                            (
                                source_col,
                                threshold,
                                4,
                            )
                        ]

                        capture_rate = result[
                            "Capture Rate"
                        ]

                        conversion_rate = result[
                            "Conversion Rate"
                        ]

                        if capture_rate is None:

                            worksheet.write_blank(
                                excel_row,
                                capture_column,
                                None,
                                capture_format,
                            )

                        else:

                            worksheet.write_number(
                                excel_row,
                                capture_column,
                                float(
                                    capture_rate
                                ),
                                capture_format,
                            )

                        if conversion_rate is None:

                            worksheet.write_blank(
                                excel_row,
                                conversion_column,
                                None,
                                conversion_format,
                            )

                        else:

                            worksheet.write_number(
                                excel_row,
                                conversion_column,
                                float(
                                    conversion_rate
                                ),
                                conversion_format,
                            )

                # =================================================
                # 5+ Anchor Percentage Matrix
                # =================================================

                current_row = (
                    percentage_first_data_row
                    + len(source_thresholds)
                    + 2
                )

                worksheet.merge_range(
                    current_row,
                    0,
                    current_row,
                    last_excel_column,
                    (
                        "IRAC / 24MOB Capture and "
                        f"Conversion Matrix | "
                        f"Anchor: {target_col} 5+"
                    ),
                    anchor_5_format,
                )

                current_row += 2

                percentage_5_period_header_row = (
                    current_row
                )

                percentage_5_subheader_row = (
                    current_row + 1
                )

                percentage_5_first_data_row = (
                    current_row + 2
                )

                worksheet.merge_range(
                    percentage_5_period_header_row,
                    0,
                    percentage_5_subheader_row,
                    0,
                    "Cut-off",
                    period_format,
                )

                for position, source_col in enumerate(
                    max_cols
                ):

                    capture_column = (
                        1 + position * 2
                    )

                    conversion_column = (
                        capture_column + 1
                    )

                    worksheet.merge_range(
                        percentage_5_period_header_row,
                        capture_column,
                        percentage_5_period_header_row,
                        conversion_column,
                        source_col,
                        period_format,
                    )

                    worksheet.write(
                        percentage_5_subheader_row,
                        capture_column,
                        "Capture",
                        subheader_format,
                    )

                    worksheet.write(
                        percentage_5_subheader_row,
                        conversion_column,
                        "Conversion",
                        subheader_format,
                    )

                for row_position, threshold in enumerate(
                    source_thresholds
                ):

                    excel_row = (
                        percentage_5_first_data_row
                        + row_position
                    )

                    worksheet.write(
                        excel_row,
                        0,
                        f"{threshold}+",
                        cutoff_format,
                    )

                    for position, source_col in enumerate(
                        max_cols
                    ):

                        capture_column = (
                            1 + position * 2
                        )

                        conversion_column = (
                            capture_column + 1
                        )

                        result = result_lookup[
                            (
                                source_col,
                                threshold,
                                5,
                            )
                        ]

                        capture_rate = result[
                            "Capture Rate"
                        ]

                        conversion_rate = result[
                            "Conversion Rate"
                        ]

                        if capture_rate is None:

                            worksheet.write_blank(
                                excel_row,
                                capture_column,
                                None,
                                capture_format,
                            )

                        else:

                            worksheet.write_number(
                                excel_row,
                                capture_column,
                                float(
                                    capture_rate
                                ),
                                capture_format,
                            )

                        if conversion_rate is None:

                            worksheet.write_blank(
                                excel_row,
                                conversion_column,
                                None,
                                conversion_format,
                            )

                        else:

                            worksheet.write_number(
                                excel_row,
                                conversion_column,
                                float(
                                    conversion_rate
                                ),
                                conversion_format,
                            )

                # =================================================
                # 4+ Value Count Matrix
                # =================================================

                current_row = (
                    percentage_5_first_data_row
                    + len(source_thresholds)
                    + 2
                )

                worksheet.merge_range(
                    current_row,
                    0,
                    current_row,
                    last_excel_column,
                    (
                        "Value Counts | "
                        f"Anchor: {target_col} 4+ | "
                        f"Target accounts: "
                        f"{total_4_target_accounts:,}"
                    ),
                    count_title_format,
                )

                current_row += 2

                count_4_period_header_row = (
                    current_row
                )

                count_4_subheader_row = (
                    current_row + 1
                )

                count_4_first_data_row = (
                    current_row + 2
                )

                worksheet.merge_range(
                    count_4_period_header_row,
                    0,
                    count_4_subheader_row,
                    0,
                    "Cut-off",
                    period_format,
                )

                for position, source_col in enumerate(
                    max_cols
                ):

                    captured_column = (
                        1 + position * 2
                    )

                    source_population_column = (
                        captured_column + 1
                    )

                    worksheet.merge_range(
                        count_4_period_header_row,
                        captured_column,
                        count_4_period_header_row,
                        source_population_column,
                        source_col,
                        period_format,
                    )

                    worksheet.write(
                        count_4_subheader_row,
                        captured_column,
                        "Captured 4+",
                        count_header_format,
                    )

                    worksheet.write(
                        count_4_subheader_row,
                        source_population_column,
                        "Source Population",
                        count_header_format,
                    )

                for row_position, threshold in enumerate(
                    source_thresholds
                ):

                    excel_row = (
                        count_4_first_data_row
                        + row_position
                    )

                    worksheet.write(
                        excel_row,
                        0,
                        f"{threshold}+",
                        cutoff_format,
                    )

                    for position, source_col in enumerate(
                        max_cols
                    ):

                        captured_column = (
                            1 + position * 2
                        )

                        source_population_column = (
                            captured_column + 1
                        )

                        result = result_lookup[
                            (
                                source_col,
                                threshold,
                                4,
                            )
                        ]

                        worksheet.write_number(
                            excel_row,
                            captured_column,
                            int(
                                result[
                                    "Captured Accounts"
                                ]
                            ),
                            count_format,
                        )

                        worksheet.write_number(
                            excel_row,
                            source_population_column,
                            int(
                                result[
                                    "Source Accounts"
                                ]
                            ),
                            count_format,
                        )

                # =================================================
                # 5+ Value Count Matrix
                # =================================================

                current_row = (
                    count_4_first_data_row
                    + len(source_thresholds)
                    + 2
                )

                worksheet.merge_range(
                    current_row,
                    0,
                    current_row,
                    last_excel_column,
                    (
                        "Value Counts | "
                        f"Anchor: {target_col} 5+ | "
                        f"Target accounts: "
                        f"{total_5_target_accounts:,}"
                    ),
                    count_title_format,
                )

                current_row += 2

                count_5_period_header_row = (
                    current_row
                )

                count_5_subheader_row = (
                    current_row + 1
                )

                count_5_first_data_row = (
                    current_row + 2
                )

                worksheet.merge_range(
                    count_5_period_header_row,
                    0,
                    count_5_subheader_row,
                    0,
                    "Cut-off",
                    period_format,
                )

                for position, source_col in enumerate(
                    max_cols
                ):

                    captured_column = (
                        1 + position * 2
                    )

                    source_population_column = (
                        captured_column + 1
                    )

                    worksheet.merge_range(
                        count_5_period_header_row,
                        captured_column,
                        count_5_period_header_row,
                        source_population_column,
                        source_col,
                        period_format,
                    )

                    worksheet.write(
                        count_5_subheader_row,
                        captured_column,
                        "Captured 5+",
                        count_header_format,
                    )

                    worksheet.write(
                        count_5_subheader_row,
                        source_population_column,
                        "Source Population",
                        count_header_format,
                    )

                for row_position, threshold in enumerate(
                    source_thresholds
                ):

                    excel_row = (
                        count_5_first_data_row
                        + row_position
                    )

                    worksheet.write(
                        excel_row,
                        0,
                        f"{threshold}+",
                        cutoff_format,
                    )

                    for position, source_col in enumerate(
                        max_cols
                    ):

                        captured_column = (
                            1 + position * 2
                        )

                        source_population_column = (
                            captured_column + 1
                        )

                        result = result_lookup[
                            (
                                source_col,
                                threshold,
                                5,
                            )
                        ]

                        worksheet.write_number(
                            excel_row,
                            captured_column,
                            int(
                                result[
                                    "Captured Accounts"
                                ]
                            ),
                            count_format,
                        )

                        worksheet.write_number(
                            excel_row,
                            source_population_column,
                            int(
                                result[
                                    "Source Accounts"
                                ]
                            ),
                            count_format,
                        )

                # -------------------------------------------------
                # Formatting / row positioning
                # -------------------------------------------------

                current_row = (
                    count_5_first_data_row
                    + len(source_thresholds)
                    + 2
                )

            # =====================================================
            # Worksheet presentation
            # =====================================================

            worksheet.set_row(0, 24)

            worksheet.set_column(
                0,
                0,
                16,
            )

            worksheet.set_column(
                1,
                last_excel_column,
                16,
            )

            worksheet.freeze_panes(
                3,
                1,
            )

    # =========================================================
    # Return combined Polars result
    # =========================================================

    if all_results:

        result_df = pl.concat(
            all_results,
            how="diagonal",
        )

    else:

        result_df = pl.DataFrame()

    return result_df
