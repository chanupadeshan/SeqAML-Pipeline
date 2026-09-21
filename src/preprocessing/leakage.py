"""
Temporal Splitting and Data Leakage Prevention
==============================================

Purpose
-------
This module splits the feature-engineered IBM AML transaction dataset into
training, validation, and test sets using chronological order.

A random split is avoided because this is a sequential transaction problem.
Random splitting could allow future transactions to appear in the training
set while earlier transactions appear in validation or test data.

The chronological split ensures that the model learns from past transactions
and is evaluated on later transactions.


Main Tasks
----------
1. Validate required columns.
2. Validate Train / Validation / Test split ratios.
3. Check that transaction timestamps do not contain null values.
4. Calculate chronological timestamp boundaries.
5. Split transactions into:
       - Train
       - Validation
       - Test
6. Verify that temporal ranges do not overlap.
7. Generate a summary of each split.


Default Split
-------------
Train       = approximately 70%
Validation  = approximately 15%
Test        = approximately 15%

The percentages may not be exactly 70/15/15 because multiple transactions
can have the same timestamp. Transactions sharing the same boundary
timestamp are kept in the same temporal partition.


Data Leakage Rule
-----------------
Past transactions -> Training
Later transactions -> Validation
Latest transactions -> Test

The target variable is not used to determine the temporal boundaries.

Scalers, encoders, class weights, and other learned preprocessing parameters
must later be fitted using the training dataset only.


Pipeline Position
-----------------
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
    +--> Train
    +--> Validation
    +--> Test
    |
    v
transform.py
    |
    v
sequences.py
    |
    v
Model Training
"""


import polars as pl

required_columns = [
    "timestamp",
    "is_laundering"
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

def _validate_split_ratios(train_ratio:float,validation_ratio:float) ->None:
    """
    Validate the train and Validation ratios.
    the remaining ratio is assigned to the test set.
    (purpose : ensure that the ratios are valid and that the sum of train and validation ratios is less than 1)
    """

    ## validation the train ratio is between 0 and 1
    if not (0 < train_ratio < 1):
        raise ValueError(f"train_ratio must be between 0 and 1. Got {train_ratio}.")
    ## validation the validation ratio is between 0 and 1
    if not (0 < validation_ratio < 1):
        raise ValueError(f"validation_ratio must be between 0 and 1. Got {validation_ratio}.")
    ## validation the sum of train and validation ratios is less than 1
    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            f"The sum of train_ratio and validation_ratio must be less than 1. "
            f"Got train_ratio={train_ratio} and validation_ratio={validation_ratio}."
        )

def _validate_timestamp_column(df:pl.LazyFrame) -> None:
    """
    ensure every transaction has a valid timestamp before temporal splitting.

    Null timestamps could cause transactions to disappear during
    chronological filtering.
    """

    null_count = df.select(pl.col("timestamp").null_count().alias("null_timestamps")).collect()["null_timestamps"][0]
    if null_count > 0:
        raise ValueError(f"Timestamp column contains {null_count} null values.")


def get_temporal_cutoffs(df:pl.LazyFrame,train_ratio:float = 0.7, validation_ratio:float = 0.15) -> tuple:
    """
        Calculate the timestamp boundaries used to split the dataset
        chronologically into training, validation, and test sets.

        Purpose
        -------
        This function does not split the dataset itself.

        Its purpose is to find the two timestamp boundaries that determine
        where the training set ends, where the validation set ends, and
        where the test set begins.

        The function uses timestamp quantiles based on the requested
        train and validation ratios.

        Example
        -------
        Suppose the dataset contains 100 transactions ordered by time.

        With:
            train_ratio = 0.70
            validation_ratio = 0.15

        the intended split is approximately:

            Transactions 1 - 70   -> Training
            Transactions 71 - 85  -> Validation
            Transactions 86 - 100 -> Test

        The function finds the timestamps around the 70% and 85%
        transaction positions.

        For example, it may calculate:

            train_end      = 2022-09-20 10:00
            validation_end = 2022-09-24 14:00

        The timeline would then look like:

            Sept 1 ------------ Sept 20 -------- Sept 24 -------- Sept 28
                                    |                 |
                                train_end       validation_end

            Training            Validation            Test

        Therefore:

            timestamp < train_end
                -> Training

            train_end <= timestamp < validation_end
                -> Validation

            timestamp >= validation_end
                -> Test

        Before calculating the cutoff timestamps, the function validates:
            - required columns
            - train and validation ratios
            - timestamp values

        Parameters
        ----------
        df : pl.LazyFrame
            Feature-engineered transaction dataset containing a valid
            `timestamp` column.

        train_ratio : float, default=0.70
            Approximate proportion of transactions assigned to the
            training set.

        validation_ratio : float, default=0.15
            Approximate proportion of transactions assigned to the
            validation set.

        Returns
        -------
        tuple
            A tuple containing:

            train_end
                Timestamp marking the end of the training period and
                the beginning of the validation period.

            validation_end
                Timestamp marking the end of the validation period and
                the beginning of the test period.

        Notes
        -----
        The function only calculates the temporal cutoff timestamps.

        The actual Train / Validation / Test datasets are created later
        by the `temporal_split()` function.
        """

    _validate_columns(df)
    _validate_split_ratios(train_ratio, validation_ratio)
    _validate_timestamp_column(df)

    validation_end_ratio = (train_ratio + validation_ratio)
    cutoffs = (
        df.select(
            [
                pl.col("timestamp").quantile(train_ratio).alias("train_end"),
                pl.col("timestamp").quantile(validation_end_ratio).alias("validation_end"),
            ]
        )
        .collect()
    )

    train_end = cutoffs["train_end"][0]
    validation_end = cutoffs["validation_end"][0]

    if train_end is None:
        raise ValueError("Unable to calculate training cutoff timestamp. Check the train_ratio and data distribution.")
    
    if validation_end is None:
        raise ValueError("Unable to calculate validation cutoff timestamp. Check the validation_ratio and data distribution.")

    if train_end >= validation_end:
        raise ValueError(
            f"Calculated train_end ({train_end}) is not less than validation_end ({validation_end}). "
            "This may indicate that the provided ratios are too high or that the data distribution is skewed."
        )

    return train_end, validation_end



def temporal_split(df:pl.LazyFrame,train_ratio:float = 0.7, validation_ratio:float = 0.15) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """
    Split the transaction dataset chronologically into training,
    validation, and test datasets.

    Purpose
    -------
    This function uses the timestamp boundaries calculated by
    `get_temporal_cutoffs()` and creates the actual Train,
    Validation, and Test datasets.

    The split is based on transaction time rather than random sampling.
    This is important for the AML sequential modeling task because
    earlier transactions should be used for training and later
    transactions should be used for validation and testing.

    Example
    -------
    Suppose `get_temporal_cutoffs()` returns:

        train_end      = 2022-09-20 10:00
        validation_end = 2022-09-24 14:00

    The dataset is then divided as:

        Sept 1 ------------ Sept 20 -------- Sept 24 -------- Sept 28
                                 |                 |
                             train_end       validation_end

        |------ Train ------|-- Validation --|------ Test ------|

    The filtering rules are:

        timestamp < train_end
            -> Training set

        train_end <= timestamp < validation_end
            -> Validation set

        timestamp >= validation_end
            -> Test set

    For example:

        2022-09-10 08:00
            -> Training

        2022-09-22 12:00
            -> Validation

        2022-09-26 16:00
            -> Test

    Parameters
    ----------
    df : pl.LazyFrame
        Feature-engineered transaction dataset containing a valid
        `timestamp` column.

    train_ratio : float, default=0.70
        Approximate proportion of transactions intended for the
        training set.

    validation_ratio : float, default=0.15
        Approximate proportion of transactions intended for the
        validation set.

    Returns
    -------
    tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]
        Returns three Polars LazyFrames:

        train_df
            Contains the earliest transactions.

        validation_df
            Contains transactions occurring after the training period
            and before the test period.

        test_df
            Contains the latest transactions.

    Notes
    -----
    This function does not independently calculate the cutoff timestamps.
    It calls `get_temporal_cutoffs()` to obtain `train_end` and
    `validation_end`.

    The main goal is to preserve chronological order and prevent future
    transactions from being mixed into the training dataset.
    """

    train_end, validation_end = get_temporal_cutoffs(df, train_ratio, validation_ratio)

    train_df = df.filter(pl.col("timestamp")< train_end)

    validation_df = df.filter(
        (pl.col("timestamp") >= train_end) 
        & (pl.col("timestamp") < validation_end)
    )

    test_df = df.filter(pl.col("timestamp") >= validation_end)

    return train_df, validation_df, test_df


def get_split_summary(train_df:pl.LazyFrame,validate_df:pl.LazyFrame,test_df:pl.LazyFrame) -> pl.DataFrame:
    """
    Generate a summary report for the Train, Validation, and Test datasets.

    Purpose
    -------
    This function does not modify or split the dataset.

    Its purpose is to inspect the three datasets created by
    `temporal_split()` and provide useful information about each split.

    The summary helps verify:
        - how many transactions are in each split
        - how many laundering transactions are in each split
        - the earliest transaction timestamp in each split
        - the latest transaction timestamp in each split

    This information is useful for checking whether the chronological
    split produced reasonable Train, Validation, and Test datasets.

    Example
    -------
    Suppose the temporal split produced:

        Train:
            22,000,000 transactions
            24,000 laundering transactions
            2022-09-01 -> 2022-09-19

        Validation:
            4,800,000 transactions
            5,000 laundering transactions
            2022-09-20 -> 2022-09-23

        Test:
            5,000,000 transactions
            6,000 laundering transactions
            2022-09-24 -> 2022-09-28

    The returned summary would look approximately like:

        split        num_transactions    num_laundering_transactions
        -------------------------------------------------------------
        train        22,000,000          24,000
        validation    4,800,000           5,000
        test          5,000,000           6,000

        split        min_timestamp       max_timestamp
        ------------------------------------------------
        train        2022-09-01          2022-09-19
        validation   2022-09-20          2022-09-23
        test         2022-09-24          2022-09-28

    Parameters
    ----------
    train_df : pl.LazyFrame
        Training dataset containing the earliest transactions.

    validate_df : pl.LazyFrame
        Validation dataset containing transactions after the
        training period.

    test_df : pl.LazyFrame
        Test dataset containing the latest transactions.

    Returns
    -------
    pl.DataFrame
        A summary table containing:

        - split
        - num_transactions
        - num_laundering_transactions
        - min_timestamp
        - max_timestamp

    Notes
    -----
    This function is mainly used for inspection and validation.

    It does not change the Train, Validation, or Test datasets.
    The returned summary is later used by `validate_temporal_split()`
    to check whether the temporal split is valid.
    """

    summary = pl.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "num_transactions": [
                train_df.select(pl.count()).collect()[0, 0],
                validate_df.select(pl.count()).collect()[0, 0],
                test_df.select(pl.count()).collect()[0, 0],
            ],
            "num_laundering_transactions": [
                train_df.filter(pl.col("is_laundering") == True).select(pl.count()).collect()[0, 0],
                validate_df.filter(pl.col("is_laundering") == True).select(pl.count()).collect()[0, 0],
                test_df.filter(pl.col("is_laundering") == True).select(pl.count()).collect()[0, 0],
            ],
            "min_timestamp": [
                train_df.select(pl.col("timestamp").min()).collect()[0, 0],
                validate_df.select(pl.col("timestamp").min()).collect()[0, 0],
                test_df.select(pl.col("timestamp").min()).collect()[0, 0],
            ],
            "max_timestamp": [
                train_df.select(pl.col("timestamp").max()).collect()[0, 0],
                validate_df.select(pl.col("timestamp").max()).collect()[0, 0],
                test_df.select(pl.col("timestamp").max()).collect()[0, 0],
            ],
        }
    )

    return summary

def validate_temporal_split(train_df:pl.LazyFrame,validate_df:pl.LazyFrame,test_df:pl.LazyFrame) -> None:
    """
    Validate that the Train, Validation, and Test datasets are correctly
    separated in chronological order.

    Purpose
    -------
    This function checks whether the temporal split created by
    `temporal_split()` is valid.

    It does not create or modify the datasets.

    Instead, it verifies that:

        1. The training set is not empty.
        2. The validation set is not empty.
        3. The test set is not empty.
        4. The training period does not overlap with the validation period.
        5. The validation period does not overlap with the test period.

    The function uses `get_split_summary()` to obtain the number of
    transactions and timestamp ranges for each split.

    Example
    -------
    A correct chronological split may look like:

        Train:
            2022-09-01 -> 2022-09-19

        Validation:
            2022-09-20 -> 2022-09-23

        Test:
            2022-09-24 -> 2022-09-28

    Timeline:

        Sept 1 -------- Sept 19   Sept 20 ----- Sept 23   Sept 24 ----- Sept 28
        |---- Train -----|        |-- Validation --|      |---- Test -----|

    This is valid because:

        train_max < validation_min

        2022-09-19 < 2022-09-20

    and:

        validation_max < test_min

        2022-09-23 < 2022-09-24

    Therefore, the three time periods do not overlap.


    Invalid Example
    ---------------
    Suppose the splits were:

        Train:
            2022-09-01 -> 2022-09-22

        Validation:
            2022-09-20 -> 2022-09-24

    In this case:

        train_max = 2022-09-22
        validation_min = 2022-09-20

    Therefore:

        train_max >= validation_min

    This means the Train and Validation periods overlap.

    The function will raise a ValueError because the temporal split
    is not valid.


    Empty Split Example
    -------------------
    If the split produces:

        Train       = 20,000,000 transactions
        Validation  = 5,000,000 transactions
        Test        = 0 transactions

    the function will raise an error because the Test dataset is empty.


    Parameters
    ----------
    train_df : pl.LazyFrame
        Training dataset containing the earliest transactions.

    validate_df : pl.LazyFrame
        Validation dataset containing transactions after the
        training period.

    test_df : pl.LazyFrame
        Test dataset containing the latest transactions.

    Returns
    -------
    None
        The function does not return a dataset.

        If the temporal split is valid, execution continues normally.

        If a problem is detected, the function raises a ValueError.

    Notes
    -----
    This validation is important because temporal overlap could allow
    later transaction periods to appear in earlier model-development
    stages, which would make the experimental setup unreliable.

    The function is called by `split_and_validate()` after the
    Train / Validation / Test datasets have been created.
    """

    summary = get_split_summary(train_df, validate_df, test_df)
    
    train_row = summary.filter(pl.col("split") == "train")
    validate_row = summary.filter(pl.col("split") == "validation")
    test_row = summary.filter(pl.col("split") == "test")

    train_count = train_row["num_transactions"][0]
    validate_count = validate_row["num_transactions"][0]
    test_count = test_row["num_transactions"][0]

    if train_count == 0:
        raise ValueError("Training dataset is empty. Check the train_ratio and data distribution.")
    if validate_count == 0:
        raise ValueError("Validation dataset is empty. Check the validation_ratio and data distribution.")
    if test_count == 0:
        raise ValueError("Test dataset is empty. Check the train_ratio and validation_ratio.")

    train_max = train_row["max_timestamp"][0]
    validate_min = validate_row["min_timestamp"][0]
    validate_max = validate_row["max_timestamp"][0]
    test_min = test_row["min_timestamp"][0]

    if train_max >= validate_min:
        raise ValueError(
            f"Training max timestamp ({train_max}) overlaps with Validation min timestamp ({validate_min})."
        )
    if validate_max >= test_min:
        raise ValueError(
            f"Validation max timestamp ({validate_max}) overlaps with Test min timestamp ({test_min})."
        )

def split_and_validate(df:pl.LazyFrame,train_ratio:float=0.7,validation_ratio:float=0.15)-> tuple[pl.LazyFrame,pl.LazyFrame,pl.LazyFrame]:
    """
    Main function for chronological dataset splitting.

    Steps
    -----
    1. Validate the dataset.
    2. Calculate temporal cutoff points.
    3. Create Train / Validation / Test splits.
    4. Validate that the time periods do not overlap.
    5. Return the three LazyFrames.
    """

    train_df, validate_df, test_df = temporal_split(df, train_ratio, validation_ratio)
    validate_temporal_split(train_df, validate_df, test_df)

    return train_df, validate_df, test_df
    
