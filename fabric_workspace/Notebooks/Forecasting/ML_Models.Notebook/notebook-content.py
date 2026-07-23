# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "22746de3-183e-4327-a844-dceda0b7165c",
# META       "default_lakehouse_name": "Sales_Forecasting",
# META       "default_lakehouse_workspace_id": "991f5e4b-c174-4ff2-992e-feb17d49d25a",
# META       "known_lakehouses": [
# META         {
# META           "id": "22746de3-183e-4327-a844-dceda0b7165c"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Welcome to your new notebook
# Typeimport numpy as np
from functools import reduce
from pyspark.sql import DataFrame, functions as F, Window
from pyspark.ml.feature import StringIndexer, VectorAssembler
from xgboost.spark import SparkXGBRegressor
import pandas as pd
from pyspark.sql.types import *
import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.types import *
import operator
from functools import reduce
import builtins


# ==========================================================
# CONFIG — adjust to your setup
# ==========================================================



## BASELINE SELECTED DRIVER DIRECTORIES
manual_features = "/lakehouse/default/Files/Driver_Analysis/Final_Feature_Selection/"
automated_features = "/lakehouse/default/Files/Automated_Driver_Analysis/"
## BASELINE output directories
parquet_dir = "abfss://991f5e4b-c174-4ff2-992e-feb17d49d25a@onelake.dfs.fabric.microsoft.com/22746de3-183e-4327-a844-dceda0b7165c/Files/Forecasting"

Topline = True
rerun_historical_forecasts = True
driver_status = "Manual_Drivers"    ## options: "No_Drivers", "Manual_Drivers", "Automated_Drivers" 




if Topline:

    actuals_table = "Sales_Forecasting.silver.topline_cutoff_data"
    initial_target_col = "Quantity"

    DRV_GRP_COLS = ['Indicator']
    ACT_GRP_COLS = ['Product_Category','series']

    ## DRIVER DIRECTORY
    if driver_status == 'Manual_Drivers':
        selected_driver_dir = manual_features + "final_features_topline.csv"
    else:
        selected_driver_dir = automated_features + "Topline/topline_xgboost_selected_feature.xlsx"


    ## OUTPUT DIRECTORIES
    parquet_dir = parquet_dir + "/Topline/"
    XGB_dir = parquet_dir + "XGBoost_Output.parquet"
else:

    actuals_table = "Sales_Forecasting.silver.middle_cutoff_data"
    initial_target_col = 'Quantity'

    DRV_GRP_COLS = ['Region','Indicator']
    ACT_GRP_COLS = ['Product_Category', 'Region','series']

    ## DRIVER DIRECTORY
    if driver_status == 'Manual_Drivers':
        selected_driver_dir = manual_features + "final_features_middle.csv"
    else:
        selected_driver_dir = automated_features + "Middle/middle_xgboost_selected_features.xlsx"


    ## OUTPUT DIRECTORIES
    parquet_dir = parquet_dir + "/Middle/"
    XGB_dir = parquet_dir + "XGBoost_Output.parquet"

# SHARED PARAMETERS


## Model Parameters
FORECAST_HORIZON = 18
SEASONAL_PERIODS = 12
MIN_TRAIN = 36 # minimum history of data before fitting (ETS model)
STEP_SIZE = 1 # move origin forward 1 months after each model iteration

## Shared Data / Pipeline Parameters
driver_table = "Sales_Forecasting.silver.compiled_drivers"
target_col = 'Value'


## SETTING DRIVER STATUS & WHICH DRIVERS TO REFERENCE
if driver_status == "No_Drivers":
    drivers_used = False
else:
    drivers_used = True


    
### Reading Data
actuals = spark.read.table(actuals_table).withColumnRenamed(initial_target_col, target_col)


def add_future_months(df, ACT_GRP_COLS, date_col='Date', horizon=18):
    """
        add future month records for each unique series
        only for actuals data without driver info
    """

    df = df.withColumn(date_col, to_date(col(date_col)))
    df = df.na.fill({'Value':0})

    # gathering the 'last date' per target series
    last_dates = (
        df.groupBy(*ACT_GRP_COLS)
        .agg(max(date_col).alias('last_date'))
    )
 

    ## Generating future horizon months
    future = (
        last_dates
        .withColumn('offset', explode(sequence(lit(1), lit(horizon))))
        .withColumn(date_col, add_months(col('last_date'), col('offset')))
        .drop('last_date','offset')
    )

    missing_cols = [c for c  in df.columns if c not in ACT_GRP_COLS + [date_col]]

    for c in missing_cols:
        future = future.withColumn(c, lit(None).cast(df.schema[c].dataType))

    # matching original schema/order
    future = future.select(df.columns)

    return df.unionByName(future)




actuals_fh_populated = add_future_months(actuals, ACT_GRP_COLS)



if drivers_used:
    ## Aggregating the drivers based upon the DRV GRP COLS defined 
    compiled_drivers = spark.read.table(driver_table)
    aggregated_drivers = compiled_drivers.groupBy(*DRV_GRP_COLS,'Date').agg(sum('Value').alias('Value'))

    # if Indicator is the only value in DRV GRP COLS then the data will be aggregated at a lower level than world
        # this ensures that "WORLD" level aggregated drivers are added to the aggregated_drivers set

    if 'Indicator' in DRV_GRP_COLS and len(DRV_GRP_COLS) > 1:
        # creates world level drivers
        world_agg = compiled_drivers.groupBy('Indicator','Date').agg(sum("Value").alias("Value"))
        world_agg = world_agg.withColumn("Indicator", concat_ws("__", col("Indicator"), lit("WORLD")))

        
        # populates the other DRV GRP COLS defined that aren't "Indicator" with the value of "World"
        for cols in DRV_GRP_COLS:
            if cols!='Indicator':
                distinct_vals = compiled_drivers.select(cols).distinct()
                world_agg = world_agg.crossJoin(distinct_vals)
                print(f"{cols} to the world agg using cross join of distinct values from the drivers data")

        aggregated_drivers = aggregated_drivers.unionByName(world_agg)
        
    else:
        print('world agg already done')



    # READING FILES W/ SELECTED DRIVERS
    ## Read file handling based on excel or csv files 
    if 'csv' in selected_driver_dir:
        print('reading csv')
        selected_drivers = spark.createDataFrame(
            pd.read_csv(selected_driver_dir)
            .drop(columns="Unnamed: 0", errors="ignore")
        )

    elif 'xlsx' in selected_driver_dir:
        print('reading excel')
        selected_drivers = spark.createDataFrame(
            pd.read_excel(selected_driver_dir)
            .drop(columns="Unnamed: 0", errors="ignore")
        )
        if Topline:
            selected_drivers = (
                selected_drivers  
                .withColumn('Lag', split(col('Feature'),"__").getItem(1))
                .withColumn('Indicator', split(col('Feature'),"__").getItem(0))
                .drop('Feature')
            )
        else:
            selected_drivers = (
                    selected_drivers
                    .withColumn('Lag', split(col("Feature"),"__").getItem(2))
                    .withColumn("Indicator", split(col("Feature"), "__").getItem(1))
                    .drop('Feature')
                )
    else:
        raise ValueError(f"Unsupported file type. expected csv or xlsx but received: {selected_driver_dir}")

    check_cols = list(dict.fromkeys(ACT_GRP_COLS+DRV_GRP_COLS))
    check_cols.remove('series')
    print(f"selected_driver num distinct records: {selected_drivers.select(*check_cols).distinct().count()}")
    print(f"agg_drivers records: {aggregated_drivers.select(*DRV_GRP_COLS).distinct().count()}")


    # JOINING DATAFRAMES

    joined_driver_data = (
        broadcast(selected_drivers)
        .join(
            aggregated_drivers,
            [*DRV_GRP_COLS],
            'left'
        )
    )

    print(f"joined records: {joined_driver_data.select(*check_cols).distinct().count()}")

    if (selected_drivers.select(*check_cols).distinct().count()) != (joined_driver_data.select(*check_cols).distinct().count()):
        raise ValueError("There is a mismatch and the number of combinations has changed post selected driver & aggregated driver join")

    ## creating a driver_date column for the join with actuals to enforce the LAG selected per driver
    joined_driver_data = (
        joined_driver_data
        .withColumn("driver_date", add_months(col('Date'), -col('Lag')))
        .drop('Date','rec_lag')
        .withColumnRenamed('Value','driver_value')
    )



    # JOINING ACTUALS AND DRIVERS 

    ## join conditions based on ACT GRP COLS w/o 'series'
    join_cols = ACT_GRP_COLS.copy()
    join_cols.remove('series')

    conditions = []
    for col_name in join_cols:
        conditions.append(
            col(f"a.{col_name}")==col(f"d.{col_name}")
        )

    ## appending date as a join condition
    conditions.append(
        col("a.Date")==col("d.driver_date")
    )

    ## creating the actual conditions
    join_cond = reduce(operator.and_, conditions)

    ## joining the actuals data with the selected drivers with their data already lagged
    actuals_w_drivers = (
        actuals_fh_populated.alias("a")
        .join(
            joined_driver_data.alias("d"),
            join_cond,
            "left"
        ).select("a.*", *[col(f"d.{c}") for c in DRV_GRP_COLS if c not in ACT_GRP_COLS], "d.driver_value", "d.Lag")
    )



    actuals_w_drivers = (
        actuals_w_drivers
        .withColumn('feature_col', concat_ws("__", *DRV_GRP_COLS, col("Lag").cast("String")))
        .drop(*DRV_GRP_COLS, "Lag")
    )

    actuals_fh_populated = actuals_w_drivers
    print('actuals_fh_populated dataframe is now overwritten with a dataframe containing driver data in long format')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## STEP 0: pivot from long to wide if drivers_used = True
def pivot_long_to_wide(sdf):
    """
    Pivots long-format driver data (feature_col, driver_value) into wide
    columns, once across ALL series. Series that don't have a given
    feature_col simply get null in that column — this is expected and
    is exactly what lets one global model share structure across series
    with different available drivers.

    Unlike pandas groupby/pivot_table, Spark's groupBy does NOT drop rows
    with a null grouping key, so target_col being null for future/extended
    rows is safe here (this was the bug in the Prophet pandas version).
    """
    if drivers_used:
        wide = (
            sdf.groupBy(*ACT_GRP_COLS,'Date',target_col)
            .pivot('feature_col')
            .agg(first('driver_value'))
        )
        return wide
    else:
        return sdf



        
# ==========================================================
# STEP 1 — calendar features
# ==========================================================
def add_calendar_features(sdf: DataFrame, date_col: str = "Date") -> DataFrame:
    sdf = sdf.withColumn("_dt",to_date(col(date_col)))
    sdf = sdf.withColumn("month",month("_dt"))
    sdf = sdf.withColumn("quarter",quarter("_dt"))
    sdf = sdf.withColumn("year",year("_dt"))
    # cyclical encodings so December/January aren't seen as "far apart"
    sdf = sdf.withColumn("month_sin",sin(col("month") * (2 * np.pi / 12)))
    sdf = sdf.withColumn("month_cos",cos(col("month") * (2 * np.pi / 12)))
    return sdf


LAGS = (1, 2, 3, 6, 12)
ROLLING_WINDOWS = (3, 6, 12)

# ==========================================================
# STEP 2 — per-series lag / rolling features 
# ==========================================================
def add_lag_rolling_features(sdf: DataFrame) -> DataFrame:
    w = Window.partitionBy(*ACT_GRP_COLS).orderBy("_dt")

    for lag_n in LAGS:
        sdf = sdf.withColumn(f"lag_{lag_n}",lag(col(target_col), lag_n).over(w))

    for win in ROLLING_WINDOWS:
        # trailing window EXCLUDING current row -> avoids leakage
        roll_w = w.rowsBetween(-win, -1)
        sdf = sdf.withColumn(f"roll_mean_{win}",avg(col(target_col)).over(roll_w))
        sdf = sdf.withColumn(f"roll_std_{win}",stddev(col(target_col)).over(roll_w))

    return sdf


# ==========================================================
# STEP 3 — build the "as-of" feature snapshot table
# One row per (series, cutoff date), containing only features knowable
# AT that cutoff (lags/rolling computed from actuals up to and including
# that date). This table is reused as the feature source for every
# horizon step, both for backtests and the final future forecast.
# ==========================================================
def build_asof_table(sdf: DataFrame) -> DataFrame:
    sdf = add_calendar_features(sdf)
    sdf = add_lag_rolling_features(sdf)
    return sdf

# ==========================================================
# STEP 4 — direct multi-horizon example builder
# For each as-of row and each horizon h in 1..FORECAST_HORIZON:
#   target_date = as-of date + h months
#   label       = target_col value AT target_date (train/backtest only)
#   drivers     = driver values AT target_date (the actual/forecasted
#                 external drivers for that future point — this is why
#                 your input must already carry forward driver values)
# ==========================================================
def build_direct_horizon_examples(
    asof_sdf: DataFrame,
    actuals_wide_sdf: DataFrame,
    driver_cols: list,
    horizon_range,
    require_label: bool,
) -> DataFrame:


    join_target = (
        actuals_wide_sdf
        .select(
            *ACT_GRP_COLS,
           to_date("Date").alias("target_date"),
           col(target_col).alias("y_target"),
            *driver_cols,
        )
    )

    panels = []
    for h in horizon_range:
        step = (
            asof_sdf
            .withColumn("Forecast_Horizon",lit(h))
            .withColumn("target_date",add_months(col("_dt"), h))
        )

        join_cond = [step[c] == join_target[c] for c in ACT_GRP_COLS] + [
            step["target_date"] == join_target["target_date"]
        ]

        joined = step.join(
            join_target.select(
                *[join_target[c].alias(f"_jt_{c}") for c in ACT_GRP_COLS],
                join_target["target_date"].alias("_jt_target_date"),
                "y_target",
                *driver_cols,
            ),
            on=[
                step[c] ==col(f"_jt_{c}") for c in ACT_GRP_COLS
            ] + [step["target_date"] ==col("_jt_target_date")],
            how="inner" if require_label else "left",
        )

        panels.append(joined)

    out = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), panels)

    if require_label:
        out = out.filter(col("y_target").isNotNull())

    return out
# ==========================================================
# STEP 5 — categorical encoding + feature assembly
# ==========================================================
def encode_and_assemble(sdf: DataFrame, driver_cols: list):
    """
    StringIndexer for group columns (ordinal encoding — fine for tree
    models). VectorAssembler with handleInvalid='keep' converts nulls
    to NaN in the feature vector rather than erroring, and XGBoost's
    native missing-value handling (missing=NaN) treats that as a real
    signal, not something to impute.
    """
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in ACT_GRP_COLS
    ]
    for idx in indexers:
        sdf = idx.fit(sdf).transform(sdf)

    feature_cols = (
        [f"{c}_idx" for c in ACT_GRP_COLS]
        + [f"lag_{l}" for l in LAGS]
        + [f"roll_mean_{w}" for w in ROLLING_WINDOWS]
        + [f"roll_std_{w}" for w in ROLLING_WINDOWS]
        + ["month", "quarter", "year", "month_sin", "month_cos", "Forecast_Horizon"]
        + driver_cols
    )

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="keep",  # nulls -> NaN in vector, not an error
    )
    sdf = assembler.transform(sdf)
    return sdf, feature_cols
# ==========================================================
# STEP 6 — train the global SparkXGBRegressor
# ==========================================================
def train_xgboost_global(train_sdf: DataFrame, num_workers: int = 8):
    xgb = SparkXGBRegressor(
        features_col="features",
        label_col="y_target",
        prediction_col="prediction",
        num_workers=num_workers,       # parallelizes training across the Spark cluster
        missing=float("nan"),          # matches VectorAssembler's handleInvalid='keep' output
        max_depth=6,
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
    )
    model = xgb.fit(train_sdf)
    return model
# ==========================================================
# MAIN ORCHESTRATOR
# ==========================================================
def fit_xgboost_global(sdf: DataFrame) -> DataFrame:
    """
    Fits ONE global XGBoost model across all series/groups, using direct
    multi-horizon features (Forecast_Horizon as a feature rather than
    recursive forecasting).

    Behavior mirrors your ARIMAX/SARIMAX/Prophet functions:
        rerun_historical_forecasts = True
            Additionally produces rolling historical backtest rows.
        rerun_historical_forecasts = False
            Skips backtests.

    Regardless of that flag, this ALWAYS produces the forward-looking
    forecast for every row where target_col is null (the 18-month
    extended window) — same guarantee as the other forecasters.
    """
    # ---- wide pivot (once, globally) ----
    sdf = pivot_long_to_wide(sdf)

    driver_cols = [c for c in sdf.columns if c not in [*ACT_GRP_COLS, "Date", target_col]]
    drivers_used = len(driver_cols) > 0
    driver_status_col =lit("Y" if drivers_used else "N")

    # ---- as-of feature snapshot (lags/rolling/calendar), built from ALL rows ----
    # (built on the full panel so both historical actuals and the future
    # extended window get calendar features; lags will naturally be null
    # wherever there's insufficient trailing history)
    
    asof_sdf = build_asof_table(sdf.select(*ACT_GRP_COLS,'Date',target_col))

    in_sample_asof = asof_sdf.filter(col(target_col).isNotNull())

    all_outputs = []

    # --------------------------------------------------
    # Optional rolling historical backtests
    # --------------------------------------------------
    if rerun_historical_forecasts:
        backtest_examples = build_direct_horizon_examples(
            in_sample_asof,
            sdf,
            driver_cols,
            horizon_range=range(1, FORECAST_HORIZON + 1),
            require_label=True,
        )
        # Only keep as-of rows with enough trailing history (MIN_TRAIN),
        # approximated here via a row-number cutoff per group.
        w = Window.partitionBy(*ACT_GRP_COLS).orderBy("_dt")
        backtest_examples = (
            backtest_examples
            .withColumn("_rn",row_number().over(w))
            .filter(col("_rn") >= MIN_TRAIN)
            .drop("_rn")
        )

        if backtest_examples.limit(1).count() > 0:
            train_bt, feature_cols = encode_and_assemble(backtest_examples, driver_cols)
            model_bt = train_xgboost_global(train_bt)
            preds_bt = model_bt.transform(train_bt)

            hist_out = preds_bt.select(
                *ACT_GRP_COLS,
               col("_dt").cast("string").alias("Training_End_Date"),
                "Forecast_Horizon",
               lit("XGBoost_Global").alias("Forecaster"),
                driver_status_col.alias("Drivers_Used_Flag"),
               col("target_date").cast("string").alias("Date"),
               col("prediction").alias("Forecast"),
               lit(None).cast("double").alias("Forecast_Lower"),
               lit(None).cast("double").alias("Forecast_Upper"),
               lit("Historical").alias("Forecast_Type"),
            )
            all_outputs.append(hist_out)

    # --------------------------------------------------
    # ALWAYS run the true forward-looking forecast:
    # use the LATEST as-of row per series (last known actual) as the
    # cutoff, and predict every horizon into the null-target rows.
    # --------------------------------------------------
    latest_w = Window.partitionBy(*ACT_GRP_COLS).orderBy(col("_dt").desc())
    latest_asof = (
        in_sample_asof
        .withColumn("_rn",row_number().over(latest_w))
        .filter(col("_rn") == 1)
        .drop("_rn")
    )

    future_examples = build_direct_horizon_examples(
        latest_asof,
        sdf,
        driver_cols,
        horizon_range=range(1, FORECAST_HORIZON + 1),
        require_label=False,   # future labels are null by definition
    )

    # Model needs to be trained on labeled history regardless — reuse the
    # same construction as the backtest panel (all in-sample horizons),
    # even if rerun_historical_forecasts was False and we didn't output it.
    training_examples = build_direct_horizon_examples(
        in_sample_asof,
        sdf,
        driver_cols,
        horizon_range=range(1, FORECAST_HORIZON + 1),
        require_label=True,
    )
    w2 = Window.partitionBy(*ACT_GRP_COLS).orderBy("_dt")
    training_examples = (
        training_examples
        .withColumn("_rn",row_number().over(w2))
        .filter(col("_rn") >= MIN_TRAIN)
        .drop("_rn")
    )

    if training_examples.limit(1).count() > 0 and future_examples.limit(1).count() > 0:
        train_final, feature_cols = encode_and_assemble(training_examples, driver_cols)
        model_final = train_xgboost_global(train_final)

        future_assembled, _ = encode_and_assemble(future_examples, driver_cols)
        preds_future = model_final.transform(future_assembled)

        future_out = preds_future.select(
            *ACT_GRP_COLS,
           col("_dt").cast("string").alias("Training_End_Date"),
            "Forecast_Horizon",
           lit("XGBoost_Global").alias("Forecaster"),
            driver_status_col.alias("Drivers_Used_Flag"),
           col("target_date").cast("string").alias("Date"),
           col("prediction").alias("Forecast"),
           lit(None).cast("double").alias("Forecast_Lower"),
           lit(None).cast("double").alias("Forecast_Upper"),
           lit("Future").alias("Forecast_Type"),
        )
        all_outputs.append(future_out)

    if len(all_outputs) == 0:
        return sdf.sparkSession.createDataFrame([], schema=None)  # empty

    return reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), all_outputs)

output = fit_xgboost_global(actuals_fh_populated.filter(col('Product_Category')=='ALU'))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
