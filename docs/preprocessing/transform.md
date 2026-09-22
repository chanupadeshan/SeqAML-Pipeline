# transform.py — Complete Notes

## 1. Purpose

`transform.py` converts the engineered transaction data into model-ready features.

Its two main jobs are:

1. **Learn preprocessing parameters from the Training dataset only.**
2. **Apply the same learned parameters to Train, Validation, Test, and later inference data.**

This is important because Validation and Test must not influence preprocessing decisions used during training.

---

## 2. Position in the preprocessing pipeline

```text
Raw CSV
   ↓
load.py
   ↓
features.py
   ↓
leakage.py
   ↓
transform.py
   ↓
processed tabular Train / Validation / Test
   ↓
sequences.py
   ↓
fixed-length sequences for LSTM / GRU
```

`transform.py` comes after feature engineering and temporal splitting.

---

## 3. Why transformation is needed

The engineered dataset contains different types of features:

```text
Numerical
- log_amount_paid
- log_amount_received
- amount_ratio
- hour
- day_of_week

Categorical
- payment_currency
- receiving_currency
- payment_format

Boolean
- is_cross_currency
- is_cross_bank
- is_same_account

Historical
- time_since_previous_transaction
```

Machine-learning models need consistent numerical inputs. Therefore the transformer:

- fills missing numerical values
- standardizes numerical features
- converts categorical values into integer IDs
- converts boolean features to compact integer values
- creates a scaled historical time-gap feature

---

## 4. Why the transformer learns from Train only

The correct workflow is:

```text
Train
   ↓
learn median / mean / std / categories
   ↓
artifacts
   ↓
apply the same artifacts to:
Train
Validation
Test
```

If mean, standard deviation, or category mappings were learned from Validation or Test, information from those datasets would leak into the training pipeline.

That problem is called **data leakage**.

The transformer therefore follows this rule:

> Fit on Train once, then reuse the learned rules everywhere else.

---

## 5. Feature groups

### Numerical features

```python
numerical_features = [
    "log_amount_paid",
    "log_amount_received",
    "amount_ratio",
    "hour",
    "day_of_week",
]
```

These features are standardized.

### Categorical features

```python
categorical_features = [
    "payment_currency",
    "receiving_currency",
    "payment_format",
]
```

These features are converted into integer category IDs.

### Boolean features

```python
boolean_features = [
    "is_cross_currency",
    "is_cross_bank",
    "is_same_account",
]
```

These are stored as `Int8`, normally `0` or `1`.

---

## 6. Required columns

The transformer expects these columns to exist:

```python
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
```

Some of these columns come from the original dataset, while others are created earlier by `features.py`.

Examples of engineered columns:

```text
sender_id
receiver_id
log_amount_paid
log_amount_received
amount_ratio
hour
day_of_week
time_since_previous_transaction
```

---

## 7. Final model feature columns

The final model-ready features are:

```python
model_feature_columns = [
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
```

The target:

```text
is_laundering
```

is **not** included in the model input feature list.

---

# 8. `_validate_columns()`

Purpose:

> Check that every column required by the transformer exists before processing begins.

The function gets the schema using:

```python
df.collect_schema().names()
```

Then it compares the available columns with `required_columns`.

If one or more required columns are missing, it raises:

```python
ValueError
```

This prevents later steps from silently producing incorrect data.

---

# 9. Historical time-gap feature

The feature:

```text
time_since_previous_transaction
```

represents the number of seconds since the previous transaction from the same sender.

Example:

```text
08:00 → first transaction → no previous gap
09:00 → second transaction → 3600 seconds
```

The first known transaction for a sender therefore has:

```text
time_since_previous_transaction = null
```

---

# 10. `has_previous_transaction`

The helper `_add_time_gap_features()` creates:

```text
has_previous_transaction
```

Logic:

```text
time gap is null      → 0
time gap is not null  → 1
```

Example:

```text
time_since_previous_transaction    has_previous_transaction
null                               0
3600                               1
7200                               1
```

This is useful because the time-gap value itself is later filled with `0`. The separate flag preserves the information that the first transaction originally had no previous transaction.

---

# 11. Log-transforming the historical time gap

The transformer creates:

```text
log_time_since_previous_transaction
```

using:

```text
log(1 + gap)
```

The code first replaces a null gap with `0`.

Why add `1`?

Because:

```text
log(0)
```

is undefined, while:

```text
log(1 + 0) = 0
```

This makes the transformation safe for first transactions.

---

# 12. Why use a log transformation?

Time gaps are usually right-skewed.

Example raw values:

```text
60
600
3600
10800
86400
```

The largest values are much larger than the smallest ones.

A log transform compresses this range while preserving the order of values.

General idea:

```text
very skewed positive values
        ↓
log transform
        ↓
smaller numerical range
```

This can make the feature easier for a model to learn from.

---

# 13. `fit_transformer()`

`fit_transformer()` learns preprocessing rules from Train only.

It does **not** perform the final transformation of all datasets.

Flow:

```text
train_df
   ↓
validate required columns
   ↓
create helper time-gap features
   ↓
prepare numerical statistics
   ↓
prepare categorical values
   ↓
execute calculations
   ↓
create artifacts dictionary
   ↓
return artifacts
```

---

# 14. The `expressions` list

Inside `fit_transformer()`:

```python
expressions = []
```

This list stores Polars expressions such as:

```python
median()
mean()
std()
unique()
sort()
```

Because the data is a `LazyFrame`, these expressions describe work to be executed later.

---

# 15. Numerical statistics

For each numerical feature, the transformer calculates:

```text
median
mean
standard deviation
```

Example:

```python
pl.col(column).median().alias(f"{column}_median")
pl.col(column).mean().alias(f"{column}_mean")
pl.col(column).std(ddof=1).alias(f"{column}_std")
```

For:

```text
log_amount_paid
```

the summary columns become:

```text
log_amount_paid_median
log_amount_paid_mean
log_amount_paid_std
```

---

# 16. Why calculate the median?

The median is used as the fallback value for missing numerical data.

Example:

```text
amount_ratio
0.98
1.00
null
1.02
```

If the Train median is:

```text
1.00
```

the missing value becomes:

```text
1.00
```

Median imputation is often robust when a feature contains extreme values.

---

# 17. Why calculate the mean?

The mean is required for standardization.

The first part of standardization is:

```text
value - mean
```

This centers the feature around zero.

---

# 18. Why calculate standard deviation?

Standard deviation represents the spread of the feature.

It is used in:

```text
scaled_value = (value - mean) / std
```

This makes numerical features more comparable in scale.

---

# 19. Standard scaling theory

The standardization formula is:

```text
z = (x - μ) / σ
```

Where:

```text
x = original value
μ = training mean
σ = training standard deviation
z = standardized value
```

Example:

```text
x = 10
mean = 8
std = 2

z = (10 - 8) / 2
z = 1
```

Interpretation:

```text
z = 0   → value is at the mean
z = 1   → one standard deviation above the mean
z = -1  → one standard deviation below the mean
```

---

# 20. `ddof=1`

The current implementation uses:

```python
.std(ddof=1)
```

This is the sample standard deviation.

Scikit-learn's `StandardScaler` uses a population-style standard deviation equivalent to:

```text
ddof=0
```

With a very large dataset, the numerical difference between `ddof=1` and `ddof=0` is extremely small.

If exact similarity to `StandardScaler` is desired, `ddof=0` is the closer choice.

---

# 21. Historical time-gap statistics

The log-transformed historical time gap is handled separately.

The transformer calculates:

```text
mean(log_time_since_previous_transaction)
std(log_time_since_previous_transaction)
```

These values are later stored in `artifacts`.

Flow:

```text
raw gap
   ↓
fill null with 0
   ↓
log(1 + gap)
   ↓
calculate Train mean and std
   ↓
store them
```

---

# 22. Categorical values

For each categorical feature, the transformer uses:

```python
drop_nulls()
unique()
sort()
implode()
```

Meaning:

```text
remove nulls
   ↓
keep unique values
   ↓
sort values
   ↓
store them as a list
```

Example:

```text
USD
EUR
USD
GBP
EUR
```

becomes:

```text
["EUR", "GBP", "USD"]
```

For `payment_currency`, the temporary summary column is:

```text
payment_currency_categories
```

This summary column is created inside the small `statistics` DataFrame, not inside the original transaction dataset.

---

# 23. `.collect()` in `fit_transformer()`

The code uses:

```python
statistics = train_df.select(expressions).collect()
```

Before `.collect()`, Polars holds the lazy query plan.

`.collect()` tells Polars to execute those calculations.

Flow:

```text
expressions
   ↓
lazy query
   ↓
.collect()
   ↓
actual statistics DataFrame
```

---

# 24. `statistics` vs `artifacts`

These are different concepts.

## `statistics`

Temporary calculated results, such as:

```text
log_amount_paid_mean
log_amount_paid_std
payment_currency_categories
```

## `artifacts`

Clean reusable preprocessing settings.

Example:

```python
{
    "numerical": {
        "log_amount_paid": {
            "fill_value": 7.5,
            "mean": 7.5,
            "std": 1.70
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
```

`statistics` is temporary.

`artifacts` is what should be reused and saved.

---

# 25. Reading previously calculated statistics

Code such as:

```python
median_value = statistics[f"{column}_median"][0]
mean_value = statistics[f"{column}_mean"][0]
std_value = statistics[f"{column}_std"][0]
```

does not calculate the values again.

It retrieves the already calculated values from the `statistics` DataFrame.

The `[0]` selects the first row because the statistics result contains one summary row.

---

# 26. Numerical safety fallbacks

The code handles missing or invalid summary statistics.

### Missing median

```python
if median_value is None:
    median_value = 0.0
```

### Missing mean

```python
if mean_value is None:
    mean_value = 0.0
```

### Missing or zero standard deviation

```python
if std_value is None or std_value == 0.0:
    std_value = 1.0
```

The last rule is especially important because division by zero would break scaling.

---

# 27. Why can standard deviation be zero?

If all values are identical:

```text
5
5
5
5
5
```

then:

```text
std = 0
```

Using a fallback of `1` gives:

```text
(5 - 5) / 1 = 0
```

which safely represents a feature with no variation.

---

# 28. Creating categorical mappings

The transformer uses:

```python
mapping = {
    str(category): index + 1
    for index, category in enumerate(categories)
}
```

Example:

```python
categories = ["EUR", "GBP", "USD"]
```

`enumerate()` gives:

```text
0, EUR
1, GBP
2, USD
```

Then `index + 1` gives:

```text
EUR → 1
GBP → 2
USD → 3
```

---

# 29. Why categories start from 1

The code reserves:

```text
0
```

for an unknown category.

Therefore known categories begin at:

```text
1
```

Example:

```text
EUR → 1
GBP → 2
USD → 3
```

If Validation contains a value never seen in Train:

```text
JPY → 0
```

---

# 30. Why Validation/Test cannot create new mappings

Suppose Train contains:

```text
USD
EUR
GBP
```

and Validation contains:

```text
JPY
```

The transformer must not learn:

```text
JPY → 4
```

from Validation.

That would modify preprocessing based on Validation data.

Correct behavior:

```text
JPY → 0
```

The mapping remains frozen after fitting on Train.

---

# 31. Final `artifacts` structure

Simplified example:

```python
{
    "numerical": {
        "log_amount_paid": {
            "fill_value": ...,
            "mean": ...,
            "std": ...
        },

        "log_amount_received": {
            "fill_value": ...,
            "mean": ...,
            "std": ...
        },

        "amount_ratio": {
            "fill_value": ...,
            "mean": ...,
            "std": ...
        },

        "hour": {
            "fill_value": ...,
            "mean": ...,
            "std": ...
        },

        "day_of_week": {
            "fill_value": ...,
            "mean": ...,
            "std": ...
        },

        "log_time_since_previous_transaction": {
            "fill_value": 0.0,
            "mean": ...,
            "std": ...
        }
    },

    "categorical_mappings": {
        "payment_currency": {
            "EUR": 1,
            "GBP": 2,
            "USD": 3
        },

        "receiving_currency": {
            ...
        },

        "payment_format": {
            ...
        }
    },

    "unknown_category_code": 0
}
```

---

# 32. `transform_dataset()`

This function applies preprocessing rules that were already learned.

It receives:

```text
dataset
+
artifacts
```

and creates transformed features.

Important:

> `transform_dataset()` must not learn new means, standard deviations, medians, or category mappings.

---

# 33. Numerical transformation

For each numerical feature:

```python
parameters = artifacts["numerical"][col]

fill_value = parameters["fill_value"]
mean_value = parameters["mean"]
std_value = parameters["std"]
```

Then the transformation is:

```text
1. fill null with Train median
2. cast to Float32
3. subtract Train mean
4. divide by Train std
```

Formula:

```text
scaled_value = (value - Train mean) / Train std
```

New column name:

```text
original_name_scaled
```

Example:

```text
log_amount_paid
      ↓
log_amount_paid_scaled
```

---

# 34. Why `Float32` is used

The current implementation casts transformed numerical features to:

```python
pl.Float32
```

Benefits:

- lower memory use than `Float64`
- suitable precision for most ML model inputs
- compatible with common PyTorch tensor types

This is useful for a dataset containing tens of millions of transactions.

---

# 35. Historical time-gap transformation

The historical feature flow is:

```text
time_since_previous_transaction
   ↓
fill missing gap with 0
   ↓
log(1 + gap)
   ↓
subtract Train log-gap mean
   ↓
divide by Train log-gap std
   ↓
log_time_since_previous_transaction_scaled
```

The separate:

```text
has_previous_transaction
```

flag is preserved.

---

# 36. Categorical encoding in `transform_dataset()`

The current implementation uses:

```python
replace_strict(
    mapping,
    default=artifacts["unknown_category_code"],
    return_dtype=pl.Int32,
)
```

Known value:

```text
USD
 ↓
mapping
 ↓
3
```

Unknown value:

```text
JPY
 ↓
not found
 ↓
0
```

This gives deterministic encoding.

---

# 37. Boolean conversion

Boolean features are cast to:

```python
pl.Int8
```

Example:

```text
is_cross_currency
is_cross_bank
is_same_account
```

They remain compact `0/1` model features.

---

# 38. Why `with_columns()` is important

The final transformer uses:

```python
df.with_columns(transformed_expressions)
```

This keeps the original columns and adds transformed columns.

Therefore important metadata remains available:

```text
timestamp
sender_id
receiver_id
is_laundering
```

This matters because `sequences.py` later needs sender identity, chronological order, and labels.

Using only:

```python
df.select(transformed_expressions)
```

would discard original columns unless they were explicitly selected.

---

# 39. Why not use `StandardScaler` directly?

`StandardScaler` is not wrong.

The reason for implementing the scaling logic in Polars is the architecture of this project:

```text
very large dataset
+
Polars LazyFrame pipeline
```

Scikit-learn's `StandardScaler` normally works with eager in-memory array/dataframe-like inputs.

Using it would usually require materializing or converting data before scaling.

The current design instead stays inside Polars:

```text
LazyFrame
   ↓
calculate Train mean/std
   ↓
scale using Polars expressions
   ↓
remain in the Polars pipeline
```

Advantages:

- less library conversion
- lower unnecessary memory overhead
- works naturally with Polars lazy execution
- explicit control over Train-only fitting
- preprocessing parameters are easy to save as custom artifacts

---

# 40. Is the custom code still standard scaling?

Yes.

The core formula is still:

```text
(value - mean) / std
```

So the theory is standardization.

The difference is only the implementation:

```text
StandardScaler
→ scikit-learn implementation

current project
→ Polars implementation
```

---

# 41. Why dataset size matters

For a small dataset, using:

```python
StandardScaler()
```

would be simple and reasonable.

For the AML dataset with tens of millions of rows, keeping transformations in Polars is more suitable because the rest of the preprocessing pipeline already uses Polars and lazy execution.

---

# 42. `fit_and_transform_splits()`

This function performs:

```text
Train
   ↓
fit_transformer()
   ↓
artifacts

artifacts + Train       → transformed Train
artifacts + Validation  → transformed Validation
artifacts + Test        → transformed Test
```

The important point is:

```text
fit_transformer()
```

is called only on Train.

---

# 43. `get_model_feature_columns()`

This function returns:

```python
model_feature_columns.copy()
```

Using `.copy()` prevents external code from accidentally changing the original module-level feature list.

---

# 44. Saving transformer artifacts

`save_transformer_artifacts()` stores the learned preprocessing configuration in JSON.

Example:

```text
artifacts/preprocessing/transformer.json
```

Why save it?

Because the exact preprocessing used during model training must also be used during:

```text
validation
testing
future inference
deployment
```

---

# 45. Loading transformer artifacts

`load_transformer_artifacts()` loads the JSON configuration.

Before opening the file, it checks:

```text
does the path exist?
is the path a file?
```

If not, it raises `FileNotFoundError`.

---

# 46. Reproducibility

Suppose the model was trained with:

```text
EUR → 1
GBP → 2
USD → 3
```

Later inference must use exactly the same mapping.

If it used:

```text
USD → 1
EUR → 2
GBP → 3
```

the model would interpret the category IDs incorrectly.

Therefore the trained model and transformer artifacts belong together.

---

# 47. Training, evaluation, and inference flow

### Training

```text
Train
 ↓
fit transformer
 ↓
save artifacts
 ↓
transform Train
 ↓
train model
```

### Validation/Test

```text
same Train artifacts
 ↓
transform Validation/Test
 ↓
evaluate model
```

### Future inference

```text
new transactions
 ↓
load saved artifacts
 ↓
apply same preprocessing
 ↓
trained model
```

The transformer should not be refitted on each new prediction batch.

---

# 48. XGBoost vs LSTM/GRU

The output of `transform.py` is already suitable as processed tabular data for an XGBoost baseline.

For LSTM and GRU:

```text
transformed transactions
   ↓
sequences.py
   ↓
ordered fixed-length sequences
```

Conceptually:

```text
XGBoost
→ one transaction represented as one tabular row

LSTM / GRU
→ multiple chronologically ordered transactions represented as a sequence
```

---

# 49. Important note about categorical IDs

Values such as:

```text
EUR → 1
GBP → 2
USD → 3
```

are category identifiers.

They do not mean:

```text
USD > GBP > EUR
```

For neural networks, a later model design can use these IDs with **embedding layers** rather than interpret them as ordinary continuous numbers.

---

# 50. `hour` and `day_of_week`

The current implementation standardizes:

```text
hour
day_of_week
```

as normal numerical features.

This is a valid simple baseline.

However, they are cyclic:

```text
23:00 is close to 00:00
Sunday is close to Monday
```

A future experiment could use sine/cosine cyclical encoding.

That is an optional improvement, not required for the current baseline.

---

# 51. `amount_ratio`

`amount_ratio` may contain extreme values.

Because standard scaling depends on mean and standard deviation, extreme values can influence the scaling.

Possible future alternatives include:

```text
log transform
clipping
robust scaling
```

Any change should be justified by EDA and model validation rather than applied automatically.

---

# 52. Data leakage checklist

```text
✓ numerical statistics learned from Train only
✓ categories learned from Train only
✓ Validation uses Train artifacts
✓ Test uses Train artifacts
✓ unknown categories map to 0
✓ target is not used as an input feature
✓ mappings remain frozen after fitting
```

---

# 53. Pytest testing strategy

Important automated tests include:

```text
fit_transformer creates expected artifacts
model feature columns exist
row count does not change
unknown categories become 0
first transactions get has_previous_transaction = 0
model-ready columns contain no nulls
Train / Validation / Test all transform successfully
```

Run normally with:

```bash
pytest tests/test_transform.py -v
```

To also display printed input/output tables:

```bash
pytest tests/test_transform.py -v -s
```

In the current environment, an unrelated pytest plugin conflict was avoided with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_transform.py -v -s
```

---

# 54. Current test result

The current transform test suite completed successfully:

```text
8 passed
```

The tests confirmed:

```text
✓ artifacts generated correctly
✓ numerical features transformed
✓ categorical features encoded
✓ JPY → 0 when unseen in Train
✓ Card → 0 when unseen in Train
✓ expected model features exist
✓ row count is preserved
✓ has_previous_transaction works
✓ model features contain no nulls
✓ all three splits transform successfully
```

---

# 55. Why row-count testing matters

Transformation should change feature values, not remove or duplicate transactions.

Expected:

```text
input row count = output row count
```

A mismatch could indicate unwanted filtering, joins, aggregation, or duplication.

---

# 56. Why null checking matters

Unexpected nulls in model features can lead to:

```text
training errors
NaN losses
invalid tensors
unreliable predictions
```

The transformed model feature set should therefore be checked before model training.

---

# 57. Deterministic preprocessing

Given:

```text
same input
+
same artifacts
```

the transformer should always produce:

```text
same output
```

This is essential for:

```text
reproducibility
debugging
validation
deployment
```

---

# 58. Performance and Polars LazyFrame

The project uses `pl.LazyFrame`.

Lazy execution allows Polars to build and optimize the query plan before executing it.

Benefits include:

```text
query optimization
efficient columnar operations
less unnecessary materialization
better handling of large datasets
```

However, `.collect()` still materializes results, so it should be used intentionally.

In `fit_transformer()`, `.collect()` is used to obtain a small summary of training statistics.

---

# 59. Model-preprocessing contract

After a model is trained, its preprocessing rules must remain stable.

The model expects exact feature meanings such as:

```text
log_amount_paid_scaled
payment_currency_encoded
has_previous_transaction
```

Therefore these should be versioned together:

```text
model
transformer artifacts
feature definitions
dataset version
```

---

# 60. Relation to MLOps

### DVC

Answers:

> Which dataset version was used?

### Transformer artifacts

Answer:

> Which preprocessing parameters and mappings were used?

### MLflow

Can later track:

```text
model
parameters
metrics
transformer.json
feature configuration
```

Together, these improve reproducibility.

---

# 61. Common mistakes to avoid

1. Fitting preprocessing on Validation.
2. Fitting preprocessing on Test.
3. Rebuilding category mappings after training.
4. Including `is_laundering` in the model input features.
5. Removing `sender_id` or `timestamp` before sequence creation.
6. Forgetting to save preprocessing artifacts.
7. Dividing by zero when standard deviation is zero.
8. Treating unseen categories inconsistently.
9. Using different preprocessing during inference.
10. Changing feature definitions after a model has already been trained.

---

# 62. Complete conceptual diagram

```text
                        TRAIN
                          │
                          ▼
                 fit_transformer()
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
       median          mean/std       categories
          │               │                │
          └───────────────┼────────────────┘
                          ▼
                      artifacts
                          │
           ┌──────────────┼──────────────┐
           │              │              │
           ▼              ▼              ▼
         Train        Validation        Test
           │              │              │
           └──────────────┼──────────────┘
                          ▼
                  transform_dataset()
                          │
                          ▼
              model-ready transactions
                          │
             ┌────────────┴─────────────┐
             ▼                          ▼
          XGBoost                  sequences.py
                                        │
                                        ▼
                                   LSTM / GRU
```

---

# 63. Function summary

| Function | Purpose |
|---|---|
| `_validate_columns()` | Check required input columns |
| `_add_time_gap_features()` | Create previous-transaction flag and log time gap |
| `fit_transformer()` | Learn preprocessing parameters from Train only |
| `transform_dataset()` | Apply learned parameters |
| `fit_and_transform_splits()` | Fit on Train and transform all three splits |
| `get_model_feature_columns()` | Return model input feature names |
| `save_transformer_artifacts()` | Save preprocessing configuration |
| `load_transformer_artifacts()` | Load saved preprocessing configuration |

---

# 64. Key terminology

## Transformation

Converting data into a form suitable for model training.

## Scaling

Changing the numerical scale of a feature.

## Standardization

Applying:

```text
(x - mean) / std
```

## Encoding

Converting categorical values into numerical representations.

## Imputation

Replacing missing values using a defined rule.

## Artifact

A saved output of the ML pipeline required later.

Example:

```text
transformer.json
```

## Data leakage

Using information during model development that should not have been available at that stage.

## LazyFrame

A Polars object that stores a query plan and delays execution.

## `.collect()`

Execute the lazy Polars query.

## Categorical mapping

A dictionary such as:

```text
USD → 3
```

## Unknown category

A category that was not present when fitting on Train.

Example:

```text
Train: USD, EUR, GBP
Validation: JPY

JPY → 0
```

---

# 65. Short explanation for presentation/interview

> `transform.py` fits preprocessing parameters only on the training split to prevent data leakage. It learns numerical imputation and scaling statistics, creates stable categorical mappings with an unknown-category code, stores those settings as reusable artifacts, and applies the same rules consistently to Train, Validation, Test, and later inference data. The implementation uses Polars so it fits naturally into the large lazy-data pipeline.

---

# 66. Final takeaway

The main principle of `transform.py` is:

```text
LEARN ONCE FROM TRAIN
        ↓
SAVE THE RULES
        ↓
REUSE THE SAME RULES EVERYWHERE
```

This gives the project:

```text
consistent preprocessing
reduced leakage risk
reproducibility
Polars-based large-data processing
safe unknown-category handling
model-ready features
```

After `transform.py`, the next preprocessing stage is:

```text
sequences.py
```

which converts transformed transactions into ordered fixed-length sequences for LSTM and GRU models.
