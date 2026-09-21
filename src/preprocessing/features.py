"""
Feature Engineering Module for the IBM AML Transaction Dataset
===============================================================

Purpose
-------
This module is responsible for creating model-ready features from the
standardized transaction data produced by `load.py`.

The main goal of this file is to transform the original transaction
attributes into useful behavioral, transactional, numerical, and temporal
features that can later be used by machine-learning and sequential models
such as XGBoost, LSTM, and GRU.

The feature-engineering process is designed to preserve transaction history
and avoid the use of future information when creating historical features.


Main Responsibilities
---------------------
This module performs the following tasks:

1. Validate Input Columns
   - Confirms that all required transaction columns are available.
   - Raises an error if any required column is missing.

2. Create Account Identifiers
   - Creates a unique `sender_id`.
   - Creates a unique `receiver_id`.
   - Combines the bank ID and account number because identical account
     numbers may exist in different banks.

3. Create Transaction Relationship Features
   - `is_cross_currency`
       Indicates whether the payment currency and receiving currency differ.

   - `is_cross_bank`
       Indicates whether the sender and receiver belong to different banks.

   - `is_same_account`
       Indicates whether the sender and receiver represent the same
       bank-account identity.

4. Create Amount-Based Features
   - `log_amount_paid`
       Applies log transformation to the amount paid.

   - `log_amount_received`
       Applies log transformation to the amount received.

   - `amount_ratio`
       Calculates the ratio between amount paid and amount received.

   Log transformations are useful because transaction amounts were found
   during EDA to have a highly right-skewed distribution with extreme values.

5. Create Temporal Features
   - `hour`
       Extracts the hour of the transaction from 0 to 23.

   - `day_of_week`
       Extracts the ISO weekday from the timestamp:
       Monday = 1 and Sunday = 7.

6. Create Historical Features
   - `time_since_previous_transaction`
       Calculates the number of seconds since the previous transaction
       made by the same sender.

   Transactions are first sorted by `sender_id` and `timestamp` so that
   historical features follow the correct chronological order.


Functions
---------
_validate_columns(df)
    Checks whether all required input columns exist.

add_account_ids(df)
    Creates unique sender and receiver identifiers.

add_transaction_flags(df)
    Creates Boolean transaction relationship features.

add_amount_features(df)
    Creates log-transformed amount features and the amount ratio.

add_time_features(df)
    Extracts hour and day-of-week information from the timestamp.

add_historical_features(df)
    Creates historical features using previous transactions belonging
    to the same sender.

add_features(df, include_history=True)
    Main pipeline function that applies all feature-engineering steps
    in the correct order.


Input
-----
The module expects a Polars LazyFrame produced by `load_transactions()`
from `load.py`.

Expected standardized columns include:

    timestamp
    from_bank
    from_account
    to_bank
    to_account
    amount_received
    receiving_currency
    amount_paid
    payment_currency
    payment_format
    is_laundering


Output
------
The module returns a Polars LazyFrame containing the original transaction
columns together with the newly engineered features.

The additional features include:

    sender_id
    receiver_id
    is_cross_currency
    is_cross_bank
    is_same_account
    log_amount_paid
    log_amount_received
    amount_ratio
    hour
    day_of_week
    time_since_previous_transaction


Data Leakage Consideration
--------------------------
Historical features must only use information available at or before the
current transaction.

No future transaction should be used when calculating a feature for an
earlier transaction.

This requirement is especially important for the sequential AML models,
because using future information would create data leakage and produce
unrealistically high model performance.


Pipeline Position
-----------------
This module is used after data loading and before dataset splitting,
transformation, sequence construction, and model training.

Overall pipeline:

    Raw CSV
        |
        v
    load.py
        |
        v
    features.py
        |
        v
    leakage.py
        |
        v
    transform.py
        |
        v
    sequences.py
        |
        v
    Model Training
        |
        +--> XGBoost
        +--> LSTM
        +--> GRU
"""
import polars as pl

required_columns = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]


def _validate_columns(df:pl.LazyFrame) -> None:
    """
    validate the columns of the dataframe and raise an error if any required column is missing.
    """
    
    available_columns = set(df.collect_schema().names())
    missing_columns = [
        col for col in required_columns if col not in available_columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {available_columns}"
        )

def add_account_ids(df:pl.LazyFrame) -> pl.LazyFrame:
    """
    Create unique account ids for the sender and receiver 
    Bank ID is included because the same accunt number may exist at different banks.
    """

    return df.with_columns(
        [
            ## Create unique account ids for the sender
            pl.concat_str(
                [
                    pl.col("from_bank").cast(pl.String),
                    pl.col("from_account"),
                ],
                separator="::"
            ).alias("sender_id"),

            ## Create unique account ids for the receiver
            pl.concat_str(
                [
                    pl.col("to_bank").cast(pl.String),
                    pl.col("to_account"),
                ],
                separator="::"
            ).alias("receiver_id")
        ]
    )

def add_transaction_flags(df:pl.LazyFrame) -> pl.LazyFrame:
    """
    create Boolean transaction flags for the following:
    - is_cross_currency: True if the payment currency is different from the receiving currency
    - is_cross_bank: True if the sender bank is different from the receiver bank
    - is_same_account: True if the sender and receiver are the same account
    """

    return df.with_columns(
        [
            (
                pl.col("payment_currency") != pl.col("receiving_currency")
            ).alias("is_cross_currency"),

            (
                pl.col("from_bank") != pl.col("to_bank")
            ).alias("is_cross_bank"),

            (
                
                (pl.col("from_bank") == pl.col("to_bank")) & ((pl.col("from_account") == pl.col("to_account")))
                
            ).alias("is_same_account")
        ]
    )

def add_amount_features(df:pl.LazyFrame) -> pl.LazyFrame:
    """
    Create numerical transaction amount features for the following:
    - log_amount_paid: log of the amount paid
    - log_amount_received: log of the amount received
    - amount_ratio: ratio of the amount paid to the amount received

    (purpose : reduce the effect of extremely large transaction values and right-skewed distributions
     because in EDA part we found that the transaction amounts have a highly right-skewed distribution with extreme values)
    """

    return df.with_columns(
        [
            (
                pl.col("amount_paid") + 1
            ).log().alias("log_amount_paid"),
            (
                pl.col("amount_received") + 1
            ).log().alias("log_amount_received"),

            pl.when(pl.col("amount_received") != 0).then(pl.col("amount_paid") / pl.col("amount_received")).otherwise(None).alias("amount_ratio")
        ]
    )

def add_time_features(df:pl.LazyFrame) -> pl.LazyFrame:
    """
    Extract features from the transaction timestamp, including:
    - hour: the hour of the day (0-23)
    - day_of_week: ISO weekday (1=Monday, 7=Sunday)

    (purpose : Transaction timing can be part of behavioral patterns. For example, an account that normally operates during daytime hours but suddenly performs unusual transactions late at night may provide useful information to the model.
    It does not mean that a nighttime transaction is automatically money laundering. It is simply another feature the model can combine with other evidence.)
    """

    return df.with_columns(
        [
            pl.col("timestamp").dt.hour().alias("hour"),
            pl.col("timestamp").dt.weekday().alias("day_of_week"),
        ]
    )

def add_historical_features(df:pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate the features based only on previous transactions, including:
    - time_since_previous_transaction: the time in seconds since the previous transaction for the same sender

    (purpose :Transaction frequency can reveal behavioral patterns. For example, if an account suddenly sends many transactions within a very short period, that rapid activity may be useful information for the model.)
    """

    df=df.sort([
        "sender_id",
        "timestamp"
    ])

    return df.with_columns(
        pl.col("timestamp").diff().over("sender_id").dt.total_seconds().alias("time_since_previous_transaction")
    )

def add_features(df:pl.LazyFrame,include_history:bool=True) -> pl.LazyFrame:
    """
    Apply the complete feature-engineering pipeline.

    Parameters
    ----------
    df : pl.LazyFrame
        LazyFrame returned by load_transactions().

    include_history : bool, default=True
        Whether to calculate historical features such as the time
        since the previous sender transaction.

    Returns
    -------
    pl.LazyFrame
        LazyFrame containing original and engineered features.
    """

    _validate_columns(df)

    df = add_account_ids(df)
    df = add_transaction_flags(df)
    df = add_amount_features(df)
    df = add_time_features(df)

    if include_history:
        df = add_historical_features(df)

    return df
    
    


    
    
    
    