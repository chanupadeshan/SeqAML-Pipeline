import sys
from pathlib import Path
from datetime import datetime

import polars as pl
import pytest


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from preprocessing.transform import (
    fit_transformer,
    transform_dataset,
    fit_and_transform_splits,
    get_model_feature_columns,
)


# ============================================================
# SAMPLE TRAIN DATA
# ============================================================

@pytest.fixture
def train_df():

    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2022, 9, 1, 8, 0),
                datetime(2022, 9, 1, 9, 0),
                datetime(2022, 9, 2, 10, 0),
                datetime(2022, 9, 2, 12, 0),
                datetime(2022, 9, 3, 15, 0),
                datetime(2022, 9, 3, 18, 0),
            ],

            "sender_id": [
                "001::A1",
                "001::A1",
                "002::A2",
                "002::A2",
                "003::A3",
                "003::A3",
            ],

            "receiver_id": [
                "010::R1",
                "011::R2",
                "012::R3",
                "013::R4",
                "014::R5",
                "015::R6",
            ],

            "is_laundering": [
                0,
                0,
                0,
                1,
                0,
                0,
            ],

            "time_since_previous_transaction": [
                None,
                3600.0,
                None,
                7200.0,
                None,
                10800.0,
            ],

            "log_amount_paid": [
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
            ],

            "log_amount_received": [
                5.1,
                6.1,
                7.1,
                8.1,
                9.1,
                10.1,
            ],

            "amount_ratio": [
                1.00,
                1.01,
                0.99,
                1.05,
                1.02,
                0.98,
            ],

            "hour": [
                8,
                9,
                10,
                12,
                15,
                18,
            ],

            "day_of_week": [
                4,
                4,
                5,
                5,
                6,
                6,
            ],

            "payment_currency": [
                "USD",
                "EUR",
                "USD",
                "GBP",
                "EUR",
                "USD",
            ],

            "receiving_currency": [
                "USD",
                "EUR",
                "EUR",
                "GBP",
                "USD",
                "USD",
            ],

            "payment_format": [
                "Wire",
                "ACH",
                "Wire",
                "Cash",
                "ACH",
                "Wire",
            ],

            "is_cross_currency": [
                0,
                0,
                1,
                0,
                1,
                0,
            ],

            "is_cross_bank": [
                1,
                1,
                1,
                0,
                1,
                1,
            ],

            "is_same_account": [
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        }
    ).lazy()


# ============================================================
# SAMPLE VALIDATION DATA
# ============================================================

@pytest.fixture
def val_df():

    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2022, 9, 4, 8, 0),
                datetime(2022, 9, 4, 12, 0),
            ],

            "sender_id": [
                "004::A4",
                "004::A4",
            ],

            "receiver_id": [
                "020::R1",
                "021::R2",
            ],

            "is_laundering": [
                0,
                1,
            ],

            "time_since_previous_transaction": [
                None,
                14400.0,
            ],

            "log_amount_paid": [
                6.5,
                11.0,
            ],

            "log_amount_received": [
                6.4,
                10.8,
            ],

            "amount_ratio": [
                1.00,
                1.15,
            ],

            "hour": [
                8,
                12,
            ],

            "day_of_week": [
                7,
                7,
            ],

            # JPY is intentionally NOT in Train
            "payment_currency": [
                "USD",
                "JPY",
            ],

            "receiving_currency": [
                "USD",
                "JPY",
            ],

            # Card is intentionally NOT in Train
            "payment_format": [
                "Wire",
                "Card",
            ],

            "is_cross_currency": [
                0,
                0,
            ],

            "is_cross_bank": [
                1,
                1,
            ],

            "is_same_account": [
                0,
                0,
            ],
        }
    ).lazy()


# ============================================================
# SAMPLE TEST DATA
# ============================================================

@pytest.fixture
def test_df():

    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2022, 9, 5, 9, 0),
                datetime(2022, 9, 5, 16, 0),
            ],

            "sender_id": [
                "005::A5",
                "005::A5",
            ],

            "receiver_id": [
                "030::R1",
                "031::R2",
            ],

            "is_laundering": [
                0,
                1,
            ],

            "time_since_previous_transaction": [
                None,
                25200.0,
            ],

            "log_amount_paid": [
                7.0,
                10.0,
            ],

            "log_amount_received": [
                6.9,
                9.9,
            ],

            "amount_ratio": [
                1.0,
                1.1,
            ],

            "hour": [
                9,
                16,
            ],

            "day_of_week": [
                1,
                1,
            ],

            "payment_currency": [
                "EUR",
                "USD",
            ],

            "receiving_currency": [
                "EUR",
                "USD",
            ],

            "payment_format": [
                "ACH",
                "Wire",
            ],

            "is_cross_currency": [
                0,
                0,
            ],

            "is_cross_bank": [
                1,
                1,
            ],

            "is_same_account": [
                0,
                0,
            ],
        }
    ).lazy()


# ============================================================
# TEST 1
# FIT TRANSFORMER
# ============================================================

def test_fit_transformer(train_df):

    artifacts = fit_transformer(train_df)

    print("\n================ TRAIN ARTIFACTS ================")

    print(artifacts)

    assert "numerical" in artifacts
    assert "categorical_mappings" in artifacts
    assert "unknown_category_code" in artifacts

    assert artifacts["unknown_category_code"] == 0


# ============================================================
# TEST 2
# TRANSFORM OUTPUT
# ============================================================

def test_transform_dataset_visual(train_df):

    artifacts = fit_transformer(train_df)

    transformed = transform_dataset(
        train_df,
        artifacts
    )

    print("\n================ INPUT DATA ================")

    print(
        train_df.select(
            [
                "log_amount_paid",
                "amount_ratio",
                "payment_currency",
                "payment_format",
                "time_since_previous_transaction",
            ]
        ).collect()
    )

    print("\n================ TRANSFORMED DATA ================")

    print(
        transformed.select(
            [
                "log_amount_paid",
                "log_amount_paid_scaled",

                "amount_ratio",
                "amount_ratio_scaled",

                "payment_currency",
                "payment_currency_encoded",

                "payment_format",
                "payment_format_encoded",

                "time_since_previous_transaction",
                "log_time_since_previous_transaction",
                "log_time_since_previous_transaction_scaled",

                "has_previous_transaction",
            ]
        ).collect()
    )


# ============================================================
# TEST 3
# EXPECTED MODEL FEATURES
# ============================================================

def test_all_model_features_exist(train_df):

    artifacts = fit_transformer(train_df)

    transformed = transform_dataset(
        train_df,
        artifacts
    )

    available_columns = set(
        transformed.collect_schema().names()
    )

    for feature in get_model_feature_columns():

        assert feature in available_columns, (
            f"Missing model feature: {feature}"
        )


# ============================================================
# TEST 4
# ROW COUNT
# ============================================================

def test_row_count_does_not_change(train_df):

    artifacts = fit_transformer(train_df)

    transformed = transform_dataset(
        train_df,
        artifacts
    )

    original_count = (
        train_df
        .select(pl.len())
        .collect()
        .item()
    )

    transformed_count = (
        transformed
        .select(pl.len())
        .collect()
        .item()
    )

    assert original_count == transformed_count


# ============================================================
# TEST 5
# UNKNOWN CATEGORIES
# ============================================================

def test_unknown_categories_become_zero(
    train_df,
    val_df,
):

    artifacts = fit_transformer(train_df)

    transformed_val = transform_dataset(
        val_df,
        artifacts
    )

    result = (
        transformed_val
        .select(
            [
                "payment_currency",
                "payment_currency_encoded",
                "payment_format",
                "payment_format_encoded",
            ]
        )
        .collect()
    )

    print("\n================ UNKNOWN CATEGORY TEST ================")

    print(result)

    jpy = result.filter(
        pl.col("payment_currency") == "JPY"
    )

    card = result.filter(
        pl.col("payment_format") == "Card"
    )

    assert (
        jpy["payment_currency_encoded"][0]
        == artifacts["unknown_category_code"]
    )

    assert (
        card["payment_format_encoded"][0]
        == artifacts["unknown_category_code"]
    )


# ============================================================
# TEST 6
# FIRST TRANSACTION FLAG
# ============================================================

def test_has_previous_transaction(train_df):

    artifacts = fit_transformer(train_df)

    transformed = (
        transform_dataset(
            train_df,
            artifacts
        )
        .select(
            [
                "time_since_previous_transaction",
                "has_previous_transaction",
            ]
        )
        .collect()
    )

    first_transactions = transformed.filter(
        pl.col(
            "time_since_previous_transaction"
        ).is_null()
    )

    assert (
        first_transactions[
            "has_previous_transaction"
        ] == 0
    ).all()


# ============================================================
# TEST 7
# NO NULL MODEL FEATURES
# ============================================================

def test_no_null_model_features(train_df):

    artifacts = fit_transformer(train_df)

    transformed = transform_dataset(
        train_df,
        artifacts
    )

    model_features = get_model_feature_columns()

    result = transformed.select(
        [
            pl.col(col)
            .null_count()
            .alias(col)

            for col in model_features
        ]
    ).collect()

    print("\n================ NULL CHECK ================")

    print(result)

    total_nulls = sum(
        result.row(0)
    )

    assert total_nulls == 0


# ============================================================
# TEST 8
# COMPLETE THREE-WAY PIPELINE
# ============================================================

def test_fit_and_transform_splits(
    train_df,
    val_df,
    test_df,
):

    (
        artifacts,
        transformed_train,
        transformed_val,
        transformed_test,
    ) = fit_and_transform_splits(
        train_df,
        val_df,
        test_df,
    )

    assert artifacts is not None

    assert (
        transformed_train
        .select(pl.len())
        .collect()
        .item()
        == 6
    )

    assert (
        transformed_val
        .select(pl.len())
        .collect()
        .item()
        == 2
    )

    assert (
        transformed_test
        .select(pl.len())
        .collect()
        .item()
        == 2
    )