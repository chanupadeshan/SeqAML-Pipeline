import json
from pathlib import Path
import polars as pl



numerical_features = [
        "log_amount_paid",
        "log_amount_received",
        "amount_ratio",
        "hour",
        "day_of_week",
]


categorical_features = [
        "payment_currency",
        "receiving_currency",
        "payment_format",
]

boolean_features = [
    "is_cross_currency",
    "is_cross_bank",
    "is_same_account",
]

required_columns = [
    "timestamp",
    "sender_id",
    "receiver_id",
    "is_laundering",
    "time_since_previous_transaction",
    *numerical_features,
    *categorical_features,
    *boolean_features,
]


model_feature_columns =[
    "log_amount_paid_scaled",
    "log_amount_received_scaled",
    "amount_ratio_scaled",
    "hour_scaled",
    "day_of_week_scaled",
    "log_time_since_previous_transaction_scaled",
    "has_previous_transaction",
    "payment_currency_encoded",
    "receiving_currency_encoded",
    "payment_format_encoded",
    "is_cross_currency",
    "is_cross_bank",
    "is_same_account",
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


def _add_time_gap_features(df:pl.LazyFrame) -> pl.LazyFrame:
    """
    Prepare the historical transaction-gap feature.

    The first transaction for each sender has no previous transaction,
    so `time_since_previous_transaction` is null.

    This function creates:

        has_previous_transaction
            1 when a previous transaction exists.
            0 when this is the first known sender transaction.

        log_time_since_previous_transaction
            log(1 + time gap in seconds).

    Null time gaps are replaced with 0 before log transformation.
    """
    
    return df.with_columns(
        [
            ## check does this transaction have a previous transaction for this same sender
            pl.col("time_since_previous_transaction").is_not_null().cast(pl.Int8).alias("has_previous_transaction"),
            ## If a previous transaction exists, its time gap is log-transformed. If no previous transaction exists, the null gap becomes 0, and log(1 + 0) = 0
            (pl.col("time_since_previous_transaction").fill_null(0).cast(pl.Float32) + 1.0).log().alias("log_time_since_previous_transaction"),
        ]
    )

def fit_transformer(train_df:pl.LazyFrame) -> dict:

    """
      
        Purpose
        -------
        This function does not transform the Train, Validation, or Test datasets.

        Its main purpose is to study the training dataset and learn the
        preprocessing settings that will later be reused by `transform_dataset()`.

        The learned settings are stored inside an `artifacts` dictionary.

        Overall flow
        ------------
            train_df
                |
                v
            fit_transformer()
                |
                v
            Learn from Train only
                |
                +--> median
                +--> mean
                +--> standard deviation
                +--> categorical mappings
                |
                v
            Save learned values
                |
                v
            artifacts

        Later:

            artifacts
                |
                +--> transform Train
                +--> transform Validation
                +--> transform Test

        This design prevents data leakage because Validation and Test data
        are never used to learn preprocessing parameters.


        Parameters Learned
        ------------------
        Numerical features:
            - median
                Used as a fallback value when a numerical feature is missing.

            - mean
                Used later for standardization.

            - standard deviation
                Used later for standardization.

        Historical time-gap feature:
            - mean of the log-transformed time gap
            - standard deviation of the log-transformed time gap

        Categorical features:
            - unique categories found in Train
            - category-to-integer mappings

        Category encoding starts from 1.

        Code 0 is reserved for:
            - unknown categories
            - categories appearing only in Validation/Test
            - missing categories


        Example
        -------
        Suppose Train contains:

            payment_currency
            ----------------
            USD
            EUR
            GBP

        The learned mapping may be:

            EUR -> 1
            GBP -> 2
            USD -> 3

        If Validation later contains:

            JPY

        JPY is not added to the mapping because the transformer must not
        learn from Validation data.

        Therefore:

            JPY -> 0


        Returns
        -------
        dict
            A dictionary containing all preprocessing parameters learned
            from the training dataset.

            Example structure:

            {
                "numerical": {
                    "log_amount_paid": {
                        "fill_value": ...,
                        "mean": ...,
                        "std": ...
                    }
                },

                "categorical_mappings": {
                    "payment_currency": {
                        "EUR": 1,
                        "GBP": 2,
                        "USD": 3
                    }
                },

                "unknown_category_code": 0
            }

        Notes
        -----
        The returned `artifacts` dictionary must be reused when transforming
        Train, Validation, and Test so that all datasets are processed using
        exactly the same rules learned from Train.

    """

    _validate_columns(train_df)
    train_df = _add_time_gap_features(train_df)

    expressions = []

    ## to collect the numerical proprocessing statistics from train only
    for column in numerical_features:
        expressions.extend(
            [
                pl.col(column).median().alias(f"{column}_median"),
                pl.col(column).mean().alias(f"{column}_mean"),
                pl.col(column).std(ddof=0).alias(f"{column}_std"),
            ]
        )

    ## learn the mean and std of the log-transformed time gap feature from train only
    expressions.extend(
        [
            pl.col("log_time_since_previous_transaction").mean().alias("log_time_since_previous_transaction_mean"),
            pl.col("log_time_since_previous_transaction").std(ddof=0).alias("log_time_since_previous_transaction_std"),
        ]
    )

    ## find all unique categorical values that exist in the training dataset only, and store them for later use when transforming Validation and Test datasets
    for column in categorical_features:
        expressions.append(
            pl.col(column).drop_nulls().unique().sort().implode().alias(f"{column}_categories")
        )

    ## execute the previous defined expressions to collect the statistics from the training dataset
    statistics = (train_df.select(expressions).collect())

    artifacts = {
        "numerical":{},
        "categorical_mappings":{},
        "unknown_category_code":0,
    }

    ## numerical patterns
    for column in numerical_features:
        median_value = statistics[f"{column}_median"][0]
        mean_value = statistics[f"{column}_mean"][0]
        std_value = statistics[f"{column}_std"][0]

        ## handle null values
        ## if the median is null, it means all values are null, so we can use 0 as a fallback value
        if median_value is None:
            median_value = 0.0

        ## if the mean is null, it means all values are null, so we can use 0 as a fallback value
        if mean_value is None:
            mean_value = 0.0

        ## if the std is null or 0, it means all values are the same, so we can use 1 as a fallback value
        if std_value is None or std_value == 0.0:
            std_value = 1.0

        artifacts["numerical"][column] = {
            "fill_value": float(median_value),
            "mean": float(mean_value),
            "std": float(std_value),
        }

    ## historical time gap statistics 
    gap_mean = statistics["log_time_since_previous_transaction_mean"][0]
    gap_std = statistics["log_time_since_previous_transaction_std"][0]

    ## handle null values
    ## if the mean is null, it means all values are null, so we can use 0 as a fallback value
    if gap_mean is None:
        gap_mean = 0.0
    
    ## if the std is null or 0, it means all values are the same, so we can use 1 as a fallback value
    if gap_std is None or gap_std == 0.0:
        gap_std = 1.0
 

    artifacts["numerical"]["log_time_since_previous_transaction"] = {
        "fill_value": 0.0,
        "mean": float(gap_mean),
        "std": float(gap_std),
    }

    ## create categorical mappings
    for column in categorical_features:
        categories = statistics[f"{column}_categories"][0]

        ## if there are no categories found in the training dataset, we can use an empty list as a fallback value
        if categories is None:
            categories = []
        
        ## convert each training category to unique integer code, starting from 1. while keeping 0 for unknown categories, missing categories, and categories that appear only in Validation/Test datasets
        mapping = {str(category): index + 1 for index, category in enumerate(categories)}

        artifacts["categorical_mappings"][column] = mapping


    """ final artifacts dictionary structure:
        {
            "numerical": {
                "log_amount_paid": {
                    "fill_value": 7.1,
                    "mean": 7.8,
                    "std": 2.3
                },

                "log_amount_received": {
                    "fill_value": 7.0,
                    "mean": 7.7,
                    "std": 2.2
                },

                "amount_ratio": {
                    "fill_value": 1.0,
                    "mean": 1.05,
                    "std": 0.40
                },

                "hour": {
                    "fill_value": 12.0,
                    "mean": 11.8,
                    "std": 6.7
                },

                "day_of_week": {
                    "fill_value": 4.0,
                    "mean": 4.1,
                    "std": 2.0
                },

                "log_time_since_previous_transaction": {
                    "fill_value": 0.0,
                    "mean": 6.5,
                    "std": 2.1
                }
            },

            "categorical_mappings": {
                "payment_currency": {
                    "EUR": 1,
                    "GBP": 2,
                    "USD": 3
                },

                "receiving_currency": {
                    "EUR": 1,
                    "GBP": 2,
                    "USD": 3
                },

                "payment_format": {
                    "ACH": 1,
                    "Cash": 2,
                    "Cheque": 3,
                    "Credit Card": 4,
                    "Wire": 5
                }
            },

            "unknown_category_code": 0
        }
    """
    
    return artifacts

def transform_dataset(df:pl.LazyFrame,artifacts:dict) -> pl.LazyFrame:

    """
        Apply previously learned transformation parameters to a dataset.

        IMPORTANT:
        This function does not learn new preprocessing parameters.

        The same `artifacts` fitted on Train must be used for:
            - Train
            - Validation
            - Test

        Returns
        -------
        pl.LazyFrame
            Original columns plus model-ready transformed features.
    """

    _validate_columns(df)
    df = _add_time_gap_features(df)
    transformed_expressions = []

    ## scale normal numerical features using the mean and std learned from the training dataset
    for col in numerical_features:
        parameters = artifacts["numerical"][col]
        fill_value = parameters["fill_value"]
        mean_value = parameters["mean"]
        std_value = parameters["std"]

        ## scaked_value = (value - mean) / std
        transformed_expressions.append(
            (
                (
                    pl.col(col).fill_null(fill_value).cast(pl.Float32)-mean_value
                ) / std_value
            ).alias(f"{col}_scaled")
        )

    ## scale the historical time-gap feature using the mean and std learned from the training dataset
    gap_parameters = artifacts["numerical"]["log_time_since_previous_transaction"]
    
    ## scaled_value = (value - mean) / std
    transformed_expressions.append(
        (
            (
            pl.col("log_time_since_previous_transaction")-gap_parameters["mean"]
            )/gap_parameters["std"]
        ).alias("log_time_since_previous_transaction_scaled")
    )

    ## Encoding categorical features using the mappings learned from the training dataset
    for col in categorical_features:
        mapping = artifacts["categorical_mappings"][col]

        ## if a category is not found in the mapping, it will be replaced with 0, which is reserved for unknown categories, missing categories, and categories that appear only in Validation/Test datasets
        transformed_expressions.append(
            pl.col(col).cast(pl.String).replace_strict(mapping,default=artifacts["unknown_category_code"],return_dtype=pl.Int32).alias(f"{col}_encoded")
        )

    ## Convert Boolean features to int8(1/0)
    for col in boolean_features:
        transformed_expressions.append(
            pl.col(col).cast(pl.Int8).alias(col)
        )

    return df.with_columns(transformed_expressions)

def fit_and_transform_splits(train_df:pl.LazyFrame,val_df:pl.LazyFrame,test_df:pl.LazyFrame) -> tuple[dict,pl.LazyFrame,pl.LazyFrame,pl.LazyFrame]:
    """
        Fit the transformer on the training dataset and transform all three splits.

        Returns
        -------
        tuple
            artifacts, transformed_train_df, transformed_val_df, transformed_test_df
    """

    artifacts = fit_transformer(train_df)
    transformed_train_df = transform_dataset(train_df,artifacts)
    transformed_val_df = transform_dataset(val_df,artifacts)
    transformed_test_df = transform_dataset(test_df,artifacts)

    return artifacts,transformed_train_df,transformed_val_df,transformed_test_df


def get_model_feature_columns() -> list[str]:
    """
        Return the list of model feature columns.
    """
    return model_feature_columns.copy()


def save_transformer_artifacts(artifacts:dict,output_path:Path) -> None:
    """
        Save the transformer artifacts to a JSON file.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)

    with open(output_path,"w") as f:
        json.dump(artifacts,f,indent=4)


def load_transformer_artifacts(input_path:Path) -> dict:
    """
        Load the transformer artifacts from a JSON file.
    """

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input_path: {input_path} does not exist.")
    if not input_path.is_file():
        raise FileNotFoundError(f"input_path: {input_path} is not a file.")

    with open(input_path,"r") as f:
        artifacts = json.load(f)

    return artifacts