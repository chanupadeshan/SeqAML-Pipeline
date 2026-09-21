"""
Data Loading and Schema Validation Module for the IBM AML Dataset
=================================================================

Purpose
-------
This module is responsible for safely loading the raw IBM AML transaction
dataset and preparing its basic structure for the rest of the preprocessing
pipeline.

The dataset is loaded using a Polars LazyFrame so that large transaction
files can be processed efficiently without loading the complete dataset
into memory immediately.

This module performs only basic loading and structural preparation.
Feature engineering is handled separately in `features.py`.


Main Responsibilities
---------------------
This module performs the following tasks:

1. Validate the Dataset File Path
   - Checks whether the specified dataset path exists.
   - Confirms that the path points to a valid file.
   - Raises an error when the dataset cannot be found.

2. Load the Dataset Lazily
   - Uses `polars.scan_csv()` instead of `read_csv()`.
   - Returns a Polars LazyFrame.
   - Avoids loading the complete dataset into memory immediately.
   - This is important because the IBM AML dataset contains millions
     of transactions.

3. Validate the Raw Dataset Schema
   - Checks whether all required IBM AML columns are present.
   - Raises an error if one or more required columns are missing.

4. Standardize Column Names
   - Converts the original IBM AML column names into consistent
     snake_case names.

   Example:

       "From Bank"          -> "from_bank"
       "Amount Paid"        -> "amount_paid"
       "Payment Currency"   -> "payment_currency"
       "Is Laundering"      -> "is_laundering"

5. Parse Transaction Timestamps
   - Converts the raw timestamp string into Polars Datetime format.
   - Datetime values are required for chronological sorting,
     temporal feature engineering, dataset splitting, and sequence
     construction.


Functions
---------
_validate_file_path(file_path)
    Validates that the dataset file exists and is a valid file.

_validate_columns(df)
    Checks that all required raw IBM AML columns are available.

load_transactions(file_path, infer_schema_length=1000)
    Main loading function.

    It performs:
        - file-path validation
        - lazy CSV loading
        - schema validation
        - column renaming
        - timestamp conversion

    It returns the standardized dataset as a Polars LazyFrame.


Input
-----
The module expects the raw IBM AML transaction CSV file containing
the following columns:

    Timestamp
    From Bank
    Account
    To Bank
    Account_duplicated_0
    Amount Received
    Receiving Currency
    Amount Paid
    Payment Currency
    Payment Format
    Is Laundering


Output
------
The module returns a Polars LazyFrame containing standardized columns:

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

The timestamp column is converted to Datetime format while the remaining
data is kept in its appropriate inferred data type.


Pipeline Position
-----------------
This module is the first stage of the preprocessing pipeline.

Overall pipeline:

    Raw IBM AML CSV
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


from pathlib import Path
import polars as pl

required_columns = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account_duplicated_0",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]


column_rename = {
    "Timestamp": "timestamp",
    "From Bank": "from_bank",
    "Account": "from_account",
    "To Bank": "to_bank",
    "Account_duplicated_0": "to_account",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "is_laundering"
}


def _validate_file_path(file_path:Path) -> Path:
    """
    validate the file path and raise an error if it does not exist or is not a file.
    """

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"file_path: {file_path} does not exist.")
    if not file_path.is_file():
        raise FileNotFoundError(f"file_path: {file_path} is not a file.")

    return file_path 


def _validate_columns(df:pl.LazyFrame) -> None:
    """
    validate the columns of the dataframe and raise an error if any required column is missing.
    """
    
    schema = df.collect_schema()
    available_columns = set(schema.names())

    missing_columns = [
        col for col in required_columns if col not in available_columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {available_columns}"
        )

def load_transactions(file_path:Path,infer_schema_length:int=1000) -> pl.LazyFrame:
    """
    Load data from a CSV file and return a LazyFrame.
    1. Validate the file path.
    2. Load the data as a LazyFrame.
    3. Validate required columns.
    4. Rename columns to snake_case.
    5. Convert the timestamp column to datetime format.
    """

    file_path = _validate_file_path(file_path)

    ## Load the dataset as a LazyFrame
    df = pl.scan_csv(file_path, infer_schema_length=infer_schema_length)

    ## Validate original dataset structure
    _validate_columns(df)

    ## Rename columns to snake_case
    df = df.rename(column_rename)

    ## Convert timestamp column to datetime format
    df = df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime,"%Y/%m/%d %H:%M",strict=False).alias("timestamp")
    )

    return df
    
    
    