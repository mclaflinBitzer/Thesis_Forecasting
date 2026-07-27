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
# META     },
# META     "environment": {
# META       "environmentId": "2448fb99-ede1-b457-4606-3f14733f8d02",
# META       "workspaceId": "00000000-0000-0000-0000-000000000000"
# META     }
# META   }
# META }

# CELL ********************

%pip install optuna

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Welcome to your new notebook
import numpy as np
from pyspark.sql import DataFrame, functions as F, Window
from pyspark.ml.feature import StringIndexer, VectorAssembler
import xgboost as xgb
import pandas as pd
from pyspark.sql.types import *
from pyspark.sql.functions import *
import operator
import builtins
from functools import reduce
from sklearn.metrics import mean_absolute_error
import optuna





# ==========================================================
# CONFIG — adjust to your setup
# ==========================================================
manual_features = "/lakehouse/default/Files/Driver_Analysis/Final_Feature_Selection/"
automated_features = "/lakehouse/default/Files/Automated_Driver_Analysis/"
parquet_dir = "abfss://991f5e4b-c174-4ff2-992e-feb17d49d25a@onelake.dfs.fabric.microsoft.com/22746de3-183e-4327-a844-dceda0b7165c/Files/Forecasting"
Topline = False
rerun_historical_forecasts = True
driver_status = "Manual_Drivers"    ## options: "No_Drivers", "Manual_Drivers", "Automated_Drivers" 
if Topline:
    actuals_table = "Sales_Forecasting.silver.topline_cutoff_data"
    initial_target_col = "Quantity"
    DRV_GRP_COLS = ['Indicator']
    ACT_GRP_COLS = ['Product_Category','series']
    if driver_status == 'Manual_Drivers':
        selected_driver_dir = manual_features + "final_features_topline.csv"
    else:
        selected_driver_dir = automated_features + "Topline/topline_xgboost_selected_feature.xlsx"
    parquet_dir = parquet_dir + "/Topline/" + driver_status + "/"
    XGB_dir = parquet_dir + "XGBoost_Output.parquet"
else:
    actuals_table = "Sales_Forecasting.silver.middle_cutoff_data"
    initial_target_col = 'Quantity'
    DRV_GRP_COLS = ['Region','Indicator']
    ACT_GRP_COLS = ['Product_Category', 'Region','series']
    if driver_status == 'Manual_Drivers':
        selected_driver_dir = manual_features + "final_features_middle.csv"
    else:
        selected_driver_dir = automated_features + "Middle/middle_xgboost_selected_features.xlsx"
    parquet_dir = parquet_dir + "/Middle/"
    XGB_dir = parquet_dir + "XGBoost_Output.parquet"
FORECAST_HORIZON = 18
SEASONAL_PERIODS = 12
MIN_TRAIN = 36
STEP_SIZE = 3  # controls how many full model retrainings happen in the
               # walk-forward backtest below. Increase if too slow —
               # does not affect the always-on future forecast.
TUNE_HOLDOUT_MONTHS = FORECAST_HORIZON
N_OPTUNA_TRIALS = 30
OPTUNA_SEED = 42


driver_table = "Sales_Forecasting.silver.compiled_drivers"
target_col = 'Value'
if driver_status == "No_Drivers":
    drivers_used = False
else:
    drivers_used = True





### Reading Data
actuals = (
    spark.read.table(actuals_table)
    .withColumnRenamed(initial_target_col, target_col)
)



def add_future_months(df, ACT_GRP_COLS, date_col='Date', horizon=18):
    """
        add future month records for each unique series
    """
    df = df.withColumn(date_col, to_date(col(date_col)))

    global_max_date = df.agg(F.max(date_col).alias('last_date')).collect()[0]['last_date']
    print(f"using global max date across all series as the forecast origin: {global_max_date}")

    last_dates = (
        df.select(*ACT_GRP_COLS).distinct()
        .withColumn('last_date', lit(global_max_date))
    )
    future = (
        last_dates
        .withColumn('offset', explode(sequence(lit(1), lit(horizon))))
        .withColumn(date_col, add_months(col('last_date'), col('offset')))
        .drop('last_date','offset')
    )
    missing_cols = [c for c in df.columns if c not in ACT_GRP_COLS + [date_col]]
    for c in missing_cols:
        future = future.withColumn(c, lit(None).cast(df.schema[c].dataType))
    future = future.select(df.columns)
    return df.unionByName(future)




actuals_fh_populated = add_future_months(actuals, ACT_GRP_COLS)



if drivers_used:
    compiled_drivers = spark.read.table(driver_table)
    aggregated_drivers = compiled_drivers.groupBy(*DRV_GRP_COLS,'Date').agg(sum('Value').alias('Value'))
    if 'Indicator' in DRV_GRP_COLS and len(DRV_GRP_COLS) > 1:
        world_agg = compiled_drivers.groupBy('Indicator','Date').agg(sum("Value").alias("Value"))
        world_agg = world_agg.withColumn("Indicator", concat_ws("__", col("Indicator"), lit("WORLD")))
        for cols in DRV_GRP_COLS:
            if cols != 'Indicator':
                distinct_vals = compiled_drivers.select(cols).distinct()
                world_agg = world_agg.crossJoin(distinct_vals)
                print(f"{cols} to the world agg using cross join of distinct values from the drivers data")
        aggregated_drivers = aggregated_drivers.unionByName(world_agg)
    else:
        print('world agg already done')
    if 'csv' in selected_driver_dir:
        print('reading csv')
        selected_drivers = spark.createDataFrame(
            pd.read_csv(selected_driver_dir).drop(columns="Unnamed: 0", errors="ignore")
        )
    elif 'xlsx' in selected_driver_dir:
        print('reading excel')
        selected_drivers = spark.createDataFrame(
            pd.read_excel(selected_driver_dir).drop(columns="Unnamed: 0", errors="ignore")
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
    joined_driver_data = (
        broadcast(selected_drivers).join(aggregated_drivers, [*DRV_GRP_COLS], 'left')
    )
    print(f"joined records: {joined_driver_data.select(*check_cols).distinct().count()}")
    if (selected_drivers.select(*check_cols).distinct().count()) != (joined_driver_data.select(*check_cols).distinct().count()):
        raise ValueError("There is a mismatch and the number of combinations has changed post selected driver & aggregated driver join")
    joined_driver_data = (
        joined_driver_data
        .withColumn("driver_date", add_months(col('Date'), -col('Lag')))
        .drop('Date','rec_lag')
        .withColumnRenamed('Value','driver_value')
    )
    join_cols = ACT_GRP_COLS.copy()
    join_cols.remove('series')
    conditions = [col(f"a.{c}") == col(f"d.{c}") for c in join_cols]
    conditions.append(col("a.Date") == col("d.driver_date"))
    join_cond = reduce(operator.and_, conditions)
    actuals_w_drivers = (
        actuals_fh_populated.alias("a")
        .join(joined_driver_data.alias("d"), join_cond, "left")
        .select("a.*", *[col(f"d.{c}") for c in DRV_GRP_COLS if c not in ACT_GRP_COLS], "d.driver_value", "d.Lag")
    )
    drop_cols = [c for c in DRV_GRP_COLS if c not in ACT_GRP_COLS]
    actuals_w_drivers = (
        actuals_w_drivers
        .withColumn('feature_col', concat_ws("__", *DRV_GRP_COLS, col("Lag").cast("String")))
        .drop(*drop_cols, "Lag")
    )
    actuals_fh_populated = actuals_w_drivers
    print('actuals_fh_populated dataframe is now overwritten with a dataframe containing driver data in long format')







## Data Transformations / Processing
def pivot_long_to_wide(sdf):
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
    sdf = sdf.withColumn("_dt", to_date(col(date_col)))
    sdf = sdf.withColumn("month", month("_dt"))
    sdf = sdf.withColumn("quarter", quarter("_dt"))
    sdf = sdf.withColumn("year", year("_dt"))
    sdf = sdf.withColumn("month_sin", sin(col("month") * (2 * np.pi / 12)))
    sdf = sdf.withColumn("month_cos", cos(col("month") * (2 * np.pi / 12)))
    return sdf
LAGS = (1, 2, 3, 6, 12,18)
MAX_LAG = builtins.max(LAGS)  # 18
ROLLING_WINDOWS = (3, 6, 12)




# ==========================================================
# STEP 2 — per-series lag / rolling features
# ==========================================================
def add_lag_rolling_features(sdf: DataFrame) -> DataFrame:
    w = Window.partitionBy(*ACT_GRP_COLS).orderBy("_dt")
    for raw_n in range(1, MAX_LAG + 1):
        sdf = sdf.withColumn(f"raw_lag_{raw_n}", lag(col(target_col), raw_n).over(w))
    for win in ROLLING_WINDOWS:
        # trailing window EXCLUDING current row -> avoids leakage.
        roll_w = w.rowsBetween(-win, -1)
        sdf = sdf.withColumn(f"roll_mean_{win}", avg(col(target_col)).over(roll_w))
        sdf = sdf.withColumn(f"roll_std_{win}", stddev(col(target_col)).over(roll_w))
    return sdf




# ==========================================================
# STEP 3 — as-of feature snapshot table
# ==========================================================
def build_asof_table(sdf: DataFrame) -> DataFrame:
    sdf = add_calendar_features(sdf)
    sdf = add_lag_rolling_features(sdf)
    return sdf




# ==========================================================
# STEP 4 — direct multi-horizon example builder
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
            *[col(f"`{c}`") for c in driver_cols],
        )
    )
    panels = []
    for h in horizon_range:
        step = (
            asof_sdf
            .withColumn("Forecast_Horizon", lit(h))
            .withColumn("target_date", add_months(col("_dt"), h))
        )

        # lag_k relative to the TARGET date, not frozen at the origin:
        #   k >  h : known history, from raw_lag_{k-h}
        #   k == h : the origin's own known actual
        #   k <  h : falls after the origin -> NULL
        for k in LAGS:
            if k > h:
                eff = k - h
                step = step.withColumn(f"lag_{k}", col(f"raw_lag_{eff}"))
            elif k == h:
                step = step.withColumn(f"lag_{k}", col(target_col))
            else:
                step = step.withColumn(f"lag_{k}", lit(None).cast("double"))

        joined = step.join(
            join_target.select(
                *[join_target[c].alias(f"_jt_{c}") for c in ACT_GRP_COLS],
                join_target["target_date"].alias("_jt_target_date"),
                "y_target",
                *[F.col(f"`{c}`") for c in driver_cols],
            ),
            on=[
                step[c] == col(f"_jt_{c}") for c in ACT_GRP_COLS
            ] + [step["target_date"] == col("_jt_target_date")],
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
def fit_encoders(full_sdf):
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep").fit(full_sdf)
        for c in ACT_GRP_COLS
    ]
    return indexers


def apply_encoding(sdf: DataFrame, indexers, driver_cols: list):
    for idx in indexers:
        sdf = idx.transform(sdf)
    feature_cols = (
        [f"{c}_idx" for c in ACT_GRP_COLS]
        + [f"lag_{l}" for l in LAGS]
        + [f"roll_mean_{w}" for w in ROLLING_WINDOWS]
        + [f"roll_std_{w}" for w in ROLLING_WINDOWS]
        + ["month", "quarter", "year", "month_sin", "month_cos", "Forecast_Horizon"]
        + driver_cols
    )
    return sdf, feature_cols




# ==========================================================
# STEP 6 — train a global XGBoost model
# ==========================================================
def train_xgboost_global(train_sdf: DataFrame, feature_cols, params):
    train_pdf = train_sdf.select(*[col(f"`{c}`") for c in feature_cols], "y_target").toPandas()
    X = train_pdf[feature_cols]
    y = train_pdf["y_target"]
    model = xgb.XGBRegressor(
        **params,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        eval_metric=['rmse', 'mae'],
    )
    model.fit(X, y)
    return model, feature_cols

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def _default_params():
    """Fallback hyperparameters if tuning can't run (e.g. insufficient data)."""
    return {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }


    
def tune_hyperparameters_optuna(
    eligible_asof: DataFrame,
    sdf: DataFrame,
    indexers,
    driver_cols: list,
    holdout_months: int,
    n_trials: int,
    seed: int,
):
    """
    Tunes hyperparameters using the SAME masking technique as the
    walk-forward backtest, rather than a post-hoc filter on the
    horizon-expanded panel — this avoids the leak where an as-of row
    before the cutoff could still produce a label whose target_date
    falls inside the holdout window.
    """
    # cutoff based on the last REAL observed actual, not the max as-of
    # date — these can differ if in_sample_asof includes rows without
    # full lag history, but the label horizon always comes from real
    # observed Values in sdf.
    max_target_date = (
        sdf.filter(col(target_col).isNotNull())
        .agg(F.max("Date").alias("max_dt")).collect()[0]["max_dt"]
    )
    cutoff = pd.Timestamp(max_target_date) - pd.DateOffset(months=holdout_months)
    cutoff_date = cutoff.date()
    print(f"Optuna tuning cutoff: holding out target_date >= {cutoff_date} as validation.")

    # ============================================================
    # >>> TRAIN: mask the SOURCE, same technique as the backtest loop.
    # Any Date >= cutoff_date is nulled out in target_col BEFORE the
    # join — so no as-of row, regardless of its own _dt, can ever pull
    # in a label from inside the holdout window. This is the fix for
    # the earlier leak: it doesn't matter which origin produced the
    # row, the label simply doesn't exist to be joined.
    # ============================================================
    masked_sdf_for_tuning = sdf.withColumn(
        target_col,
        F.when(col("Date") < lit(cutoff_date), col(target_col)).otherwise(lit(None))
    )

    train_examples = build_direct_horizon_examples(
        eligible_asof, masked_sdf_for_tuning, driver_cols,
        horizon_range=range(1, FORECAST_HORIZON + 1),
        require_label=True,
    )
    # ============================================================

    # ============================================================
    # >>> VALIDATION: build against the REAL (unmasked) sdf, then just
    # SELECT the holdout window. This isn't a leakage guard — we WANT
    # real labels here — it's purely picking which rows count as the
    # held-out test set. Since sdf only has real values up to
    # max_target_date and require_label=True drops nulls, this
    # naturally captures exactly the most recent `holdout_months` of
    # real data, no more.
    # ============================================================
    val_all = build_direct_horizon_examples(
        eligible_asof, sdf, driver_cols,
        horizon_range=range(1, FORECAST_HORIZON + 1),
        require_label=True,
    )
    val_examples = val_all.filter(col("target_date") >= lit(cutoff_date))
    # ============================================================

    train_part, feature_cols = apply_encoding(train_examples, indexers, driver_cols)
    val_part, _ = apply_encoding(val_examples, indexers, driver_cols)

    if train_part.limit(1).count() == 0 or val_part.limit(1).count() == 0:
        print("Not enough data for a real holdout split — falling back to default params.")
        return _default_params()

    train_pdf = train_part.select(*[col(f"`{c}`") for c in feature_cols], "y_target").toPandas()
    val_pdf = val_part.select(*[col(f"`{c}`") for c in feature_cols], "y_target").toPandas()

    X_train, y_train = train_pdf[feature_cols], train_pdf["y_target"]
    X_val, y_val = val_pdf[feature_cols], val_pdf["y_target"]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        model = xgb.XGBRegressor(
            **params, objective="reg:squarederror", random_state=seed, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    print(f"Optuna selected hyperparameters: {best_params}")
    print(f"Optuna best validation MAE: {study.best_value:.4f}")
    return best_params

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==========================================================
# MAIN ORCHESTRATOR
# ==========================================================
def fit_xgboost_global(sdf: DataFrame) -> DataFrame:
    """
    Fits a global XGBoost model using direct multi-horizon features.

    rerun_historical_forecasts = True:
        TRUE walk-forward backtesting — for each historical origin date
        (stepped by STEP_SIZE), a model is trained ONLY on data known
        up to that origin, using a version of the actuals table where
        y_target is masked to NULL for any date beyond that origin
        (see the "source-level masking" comment inside the loop below).
        Origin dates are aligned across ALL series (see "origin date
        selection" block) so a single global model at a given origin
        has every series already past MIN_TRAIN.

    Regardless of that flag, this ALWAYS produces the forward-looking
    forecast for every row where target_col is null.
    """
    sdf = pivot_long_to_wide(sdf)

    reserved_cols = (
        ACT_GRP_COLS
        + ["Date", target_col, "_dt", "month", "quarter", "year", "month_sin", "month_cos", "Forecast_Horizon"]
        + [f"raw_lag_{n}" for n in range(1, MAX_LAG + 1)]
        + [f"roll_mean_{w}" for w in ROLLING_WINDOWS]
        + [f"roll_std_{w}" for w in ROLLING_WINDOWS]
    )

    driver_cols = sorted([c for c in sdf.columns if c not in reserved_cols])
    duplicates = [c for c in driver_cols if c in [
        "Forecast_Horizon", "_dt", "month", "quarter", "year", "month_sin", "month_cos"
    ]]
    if duplicates:
        raise ValueError(f"Unexpected engineered columns in driver_cols: {duplicates}")


    drivers_used_local = len(driver_cols) > 0
    driver_status_col = lit("Y" if drivers_used_local else "N")
    indexers = fit_encoders(sdf)
    asof_sdf = build_asof_table(sdf.select(*ACT_GRP_COLS, 'Date', target_col))
    in_sample_asof = asof_sdf.filter(col(target_col).isNotNull())


    ## BEST PARAMS SELECTION FOR XGBOOST
    best_params = tune_hyperparameters_optuna(
        in_sample_asof,
        sdf,
        indexers,
        driver_cols,
        holdout_months = FORECAST_HORIZON,
        n_trials = N_OPTUNA_TRIALS,
        seed=42
    )

    print(best_params)

    # ============================================================
    # ORIGIN DATE SELECTION — aligns backtest origins across ALL
    # series, since one global model is trained per origin. A series
    # only counts as eligible once it has MIN_TRAIN rows of its own
    # history; the MAX of each series' own earliest-eligible date
    # ("latest_min_hist_date") becomes the shared starting point, so
    # every candidate origin has every series already past MIN_TRAIN.
    # ============================================================

    w_hist = Window.partitionBy(*ACT_GRP_COLS).orderBy("_dt")
    hist_sdf = in_sample_asof.withColumn("_hist_row", row_number().over(w_hist))

    eligible = hist_sdf.filter(col("_hist_row") >= MIN_TRAIN)

    min_serie_dates = eligible.groupBy(*ACT_GRP_COLS).agg(F.min("_dt").alias("latest_min_hist_date"))
    latest_min_hist_date = (
        min_serie_dates.agg(F.max("latest_min_hist_date").alias("latest_min_hist_date")).collect()[0]["latest_min_hist_date"]
    )
    print(f"the latest min history date across all series is {latest_min_hist_date}")

    max_known_date = in_sample_asof.agg(F.max("_dt").alias("max_dt")).collect()[0]["max_dt"]
    two_year_floor = add_months(lit(max_known_date), lit(-24))
    two_year_floor_date = (
        in_sample_asof.select(two_year_floor.alias("floor_dt")).limit(1).collect()[0]["floor_dt"]
    )

    print(f"2-year-back floor for latest_min_hist_date is {two_year_floor_date}")

    if latest_min_hist_date > two_year_floor_date:
        print(
            f"latest_min_hist_date ({latest_min_hist_date}) is less than 2 years before "
            f"the max known date — capping to {two_year_floor_date} instead."
        )
        latest_min_hist_date = two_year_floor_date
    # ============================================================

    print(f"final latest_min_hist_date used for origin selection: {latest_min_hist_date}")


    eligible = eligible.filter(col("_dt") >= lit(latest_min_hist_date)).drop("_hist_row")
    # ============================================================

    all_outputs = []

    # --------------------------------------------------
    # TRUE walk-forward historical backtests
    # --------------------------------------------------
    if rerun_historical_forecasts:
        origin_dates = [
            r["_dt"] for r in
            eligible.select("_dt").distinct().orderBy("_dt").collect()
        ][::STEP_SIZE]

        print(f"Running {len(origin_dates)} walk-forward backtest origin(s) "
              f"— retrains the global model once per origin.")

        for origin in origin_dates:
            print(f"Running the backtest at origin: {origin}")
            train_pool = in_sample_asof.filter(col("_dt") <= lit(origin))

            # ============================================================
            # >>> THE FIX (source-level masking, not a downstream filter):
            # build a version of `sdf` where target_col (the real Value)
            # is set to NULL for any Date beyond this origin, BEFORE it's
            # ever passed into build_direct_horizon_examples / joined in
            # as y_target. This means a future actual is never
            # representable as a label for this origin's training run at
            # all — it doesn't exist in the frame to be joined, rather
            # than existing and being filtered out afterward. Only
            # target_col is touched: driver_cols (including forecasted
            # future driver values) come from this same frame completely
            # unaffected, since the F.when() below only conditions
            # target_col, never any other column.
            # ============================================================
            masked_actuals_for_origin = sdf.withColumn(
                target_col,
                F.when(col("Date") <= lit(origin), col(target_col)).otherwise(lit(None))
            )
            # ============================================================

            train_examples = build_direct_horizon_examples(
                train_pool, masked_actuals_for_origin, driver_cols,
                horizon_range=range(1, FORECAST_HORIZON + 1),
                require_label=True,
            )
            # (no target_date <= origin filter needed anymore — masking
            # the source already guarantees no post-origin label can
            # ever appear in join_target/y_target for this origin's run)

            if train_examples.limit(1).count() == 0:
                continue

            train_bt, feature_cols = apply_encoding(train_examples, indexers, driver_cols)
            model_bt, feature_cols = train_xgboost_global(train_bt, feature_cols, best_params)

            origin_asof = in_sample_asof.filter(col("_dt") == lit(origin))
            # Prediction uses the REAL (unmasked) sdf — origin_examples
            # is built with require_label=False and only feature_cols are
            # ever fed to model.predict(), so forecasted future driver
            # values flow through normally here.
            origin_examples = build_direct_horizon_examples(
                origin_asof, sdf, driver_cols,
                horizon_range=range(1, FORECAST_HORIZON + 1),
                require_label=False,
            )
            if origin_examples.limit(1).count() == 0:
                continue

            origin_assembled, _ = apply_encoding(origin_examples, indexers, driver_cols)
            origin_pdf = origin_assembled.select(
                *[col(f"`{c}`") for c in feature_cols], *ACT_GRP_COLS, "_dt", "target_date"
            ).toPandas()

            if len(origin_pdf) == 0:
                continue

            origin_pdf["prediction"] = model_bt.predict(origin_pdf[feature_cols])
            origin_pdf["prediction"] = origin_pdf["prediction"].clip(lower=0)
            preds_origin = spark.createDataFrame(origin_pdf)

            hist_out = preds_origin.select(
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
    print("BACKTESTING COMPLETE")

    # --------------------------------------------------
    # ALWAYS run the true forward-looking forecast
    # --------------------------------------------------
    latest_w = Window.partitionBy(*ACT_GRP_COLS).orderBy(col("_dt").desc())
    latest_asof = (
        in_sample_asof
        .withColumn("_rn", row_number().over(latest_w))
        .filter(col("_rn") == 1)
        .drop("_rn")
    )

    future_examples = build_direct_horizon_examples(
        latest_asof, sdf, driver_cols,
        horizon_range=range(1, FORECAST_HORIZON + 1),
        require_label=False,
    )

    # No masking needed here — labels only ever come from genuinely
    # observed actuals (require_label=True filters out the null
    # future rows), so there's no future outcome to leak.
    training_examples = build_direct_horizon_examples(
        eligible, sdf, driver_cols,
        horizon_range=range(1, FORECAST_HORIZON + 1),
        require_label=True,
    )

    if training_examples.limit(1).count() > 0 and future_examples.limit(1).count() > 0:
        train_final, feature_cols = apply_encoding(training_examples, indexers, driver_cols)
        model_final, feature_cols = train_xgboost_global(train_final, feature_cols, best_params)

        future_assembled, _ = apply_encoding(future_examples, indexers, driver_cols)
        future_pdf = future_assembled.select(
            *[col(f"`{c}`") for c in feature_cols], *ACT_GRP_COLS, "_dt", "target_date"
        ).toPandas()

        future_pdf["prediction"] = model_final.predict(future_pdf[feature_cols])
        future_pdf["prediction"] = future_pdf["prediction"].clip(lower=0)
        preds_future = spark.createDataFrame(future_pdf)

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
        return sdf.sparkSession.createDataFrame([], schema=None)

    return reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), all_outputs)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

output = fit_xgboost_global(actuals_fh_populated).cache()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

if rerun_historical_forecasts:
    output.write.mode('overwrite').parquet(XGB_dir)
else:
    output.write.mode('append').parquet(XGB_dir)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

dist_prod_cat = (
    output.select("Product_Category")
          .distinct()
          .rdd.flatMap(lambda x: x)
          .collect()
)

temp = spark.read.table(actuals_table)


test = output
#spark.read.parquet(Prophet_dir).cache()

test_agg = test.groupBy(*ACT_GRP_COLS, 'Date').agg(avg('Forecast').alias('Forecast'))


test_join = (
    temp.withColumnRenamed('Quantity','Value').withColumn('series', concat_ws("__", col('series'), lit('ACT')))
    .unionByName(
        test_agg.withColumnRenamed('Forecast','Value').withColumn('series', concat_ws("__", col('series'), lit('FORCAST')))
    )
)

if Topline:
    display(test_join)
else:
    for prod_catg in dist_prod_cat:
        print(f"Visualization for all {prod_catg} and region combinations")
        display(test_join.filter(col('series').like(f"{prod_catg}%")))

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
