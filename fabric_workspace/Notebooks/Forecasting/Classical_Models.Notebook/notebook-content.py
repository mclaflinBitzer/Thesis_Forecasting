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

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import pandas as pd
import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.types import *

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## BASELINE output directories
parquet_dir = "abfss://991f5e4b-c174-4ff2-992e-feb17d49d25a@onelake.dfs.fabric.microsoft.com/22746de3-183e-4327-a844-dceda0b7165c/Files/Forecasting"

Topline = False
rerun_historical_forecasts = True
ets_run = True
arimax_run = True
sarimax_run = True
prophet_run = True
driver_status = "No_Drivers"    ## options: "No_Drivers", "Manual_Drivers", "Automated_Drivers" 

if driver_status == "No_Drivers":
    drivers_used = False
else:
    drivers_used = True

if Topline:

    actuals_table = "Sales_Forecasting.silver.topline_cutoff_data"
    initial_target_col = "Quantity"

    DRV_GRP_COLS = ['Indicator']
    ACT_GRP_COLS = ['Product_Category','series']

    ## OUTPUT DIRECTORIES
    parquet_dir = parquet_dir + "/Topline/"
    Holts_dir = parquet_dir + "No_Drivers/Holts_Output.parquet"
    Arimax_dir = parquet_dir + driver_status + "/Arimax_Output.parquet"
    Sarimax_dir = parquet_dir + driver_status + "/Sarimax_output.parquet"
    Prophet_dir = parquet_dir + driver_status + "/Prophet_output.parquet"

else:

    actuals_table = "Sales_Forecasting.silver.middle_cutoff_data"
    initial_target_col = 'Quantity'

    DRV_GRP_COLS = ['Region','Indicator']
    ACT_GRP_COLS = ['Product_Category', 'Region','series']

    ## OUTPUT DIRECTORIES
    parquet_dir = parquet_dir + "/Middle/"
    Holts_dir = parquet_dir + "No_Drivers/Holts_Output.parquet"
    Arimax_dir = parquet_dir + driver_status + "/Arimax_Output.parquet"
    Sarimax_dir = parquet_dir +  driver_status + "/Sarimax_output.parquet"
    Prophet_dir = parquet_dir + driver_status + "/Prophet_output.parquet"


# SHARED PARAMETERS

## Model Parameters
FORECAST_HORIZON = 18
SEASONAL_PERIODS = 12
MIN_TRAIN = 36 # minimum history of data before fitting (ETS model)
STEP_SIZE = 1 # move origin forward 1 months after each model iteration

## Shared Data / Pipeline Parameters
driver_table = "Sales_Forecasting.silver.compiled_drivers"
target_col = 'Value'

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

test.columns

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# actuals.columns

# display(
#     actuals
#     .groupBy(*ACT_GRP_COLS,'Date')
#     .agg(avg('Value').alias('Value'))
#     .withColumn("series", concat(col("series"),lit("_ACTUALS")))
#     .unionByName(
#         test
#         .groupBy(*ACT_GRP_COLS, 'Date')
#         .agg(avg('Forecast').alias('Value'))
#         .withColumn("series", concat(col('series'), lit("_FORECAST")))
#     )
# )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# test = spark.read.parquet(Prophet_dir)
# display(test.groupBy(*ACT_GRP_COLS,'Date').agg(avg("Forecast").alias("Forecast")))
# display(test.groupBy(*ACT_GRP_COLS).agg(max("Training_End_Date")))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

actuals = spark.read.table(actuals_table).withColumnRenamed(initial_target_col, target_col)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Adding FH periods to actuals data

# CELL ********************

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
display(actuals_fh_populated.orderBy(desc('Date')).limit(3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Prophet

# CELL ********************

from prophet import Prophet
import logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)  # Prophet is verbose by default




# ============================================================
# fit+forecast logic into a helper, with optional historical backtest loop and the always-on future forecast.
# ============================================================
def _prophet_fit_and_forecast(train, future, driver_cols, drivers_used, id_vals, driver_status):
    """
    Fits Prophet on `train` and forecasts len(future) steps aligned to `future`.
    Returns an output DataFrame, or None if fitting failed / windows too short.
    """
    if len(train) < MIN_TRAIN or len(future) < FORECAST_HORIZON:
        return None

    horizon = len(future)

    prophet_train = train.rename(columns={"Date": "ds", target_col: "y"})
    prophet_train["ds"] = pd.to_datetime(prophet_train["ds"])

    prophet_future = future.rename(columns={"Date": "ds"})
    prophet_future["ds"] = pd.to_datetime(prophet_future["ds"])

    # ============================================================
    # driver prep is now conditional on drivers_used,
    # ============================================================
    if drivers_used:
        for c in driver_cols:
            prophet_train[c] = pd.to_numeric(prophet_train[c], errors="coerce")
            prophet_future[c] = pd.to_numeric(prophet_future[c], errors="coerce")

            med = prophet_train[c].median()
            prophet_train[c] = prophet_train[c].fillna(med)
            fut_med = prophet_future[c].median()
            prophet_future[c] = prophet_future[c].fillna(fut_med)
    # ============================================================

    # ---- simple train/validation split for hyperparameter comparison ----
    val_size = 6
    if len(prophet_train) <= val_size:  # >>> CHANGE #4: guard against tiny backtest windows
        return None

    train_part = prophet_train.iloc[:-val_size]
    val_part = prophet_train.iloc[-val_size:]

    param_grid = [
        {"changepoint_prior_scale": cps, "seasonality_prior_scale": sps}
        for cps in [0.01, 0.05, 0.1, 0.5]
        for sps in [0.1, 1.0, 10.0]
    ]

    # ============================================================
    # column lists only include driver_cols when drivers_used is True, so no-driver runs don't reference
    # columns that don't exist.
    # ============================================================
    fit_cols = ["ds", "y"] + (driver_cols if drivers_used else [])
    predict_cols = ["ds"] + (driver_cols if drivers_used else [])
    # ============================================================

    best_mae, best_params = np.inf, None
    for params in param_grid:
        try:
            m = Prophet(
                changepoint_prior_scale=params["changepoint_prior_scale"],
                seasonality_prior_scale=params["seasonality_prior_scale"],
                yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
            )
            if drivers_used:  # only add regressors when drivers exist
                for c in driver_cols:
                    m.add_regressor(c)
            m.fit(train_part[fit_cols])

            val_pred = m.predict(val_part[predict_cols])
            mae = np.mean(np.abs(val_part["y"].values - val_pred["yhat"].values))
            if mae < best_mae:
                best_mae, best_params = mae, params
        except Exception:
            continue

    if best_params is None:
        return None

    # ---- refit on full train data with best params ----
    final_model = Prophet(
        changepoint_prior_scale=best_params["changepoint_prior_scale"],
        seasonality_prior_scale=best_params["seasonality_prior_scale"],
        yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
    )
    if drivers_used:  
        for c in driver_cols:
            final_model.add_regressor(c)
    final_model.fit(prophet_train[fit_cols])

    forecast = final_model.predict(prophet_future[predict_cols])

    out = pd.DataFrame({
        **{c: id_vals[c] for c in ACT_GRP_COLS},
        "Training_End_Date": train["Date"].iloc[-1],   
        "Forecast_Horizon": range(1, horizon + 1),
        "Forecaster": "Prophet",                        
        "Drivers_Used_Flag": driver_status,              
        "Date": prophet_future["ds"].astype(str).values,
        "Forecast": forecast["yhat"].values,
        "Forecast_Lower": forecast["yhat_lower"].values,
        "Forecast_Upper": forecast["yhat_upper"].values,
        "best_changepoint_prior_scale": best_params["changepoint_prior_scale"],
        "best_seasonality_prior_scale": best_params["seasonality_prior_scale"],
        "cv_mae": best_mae,
    })
    return out


def fit_prophet(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Fits Prophet with or without external driver regressors.

    Behavior:
        rerun_historical_forecasts = True
            Additionally performs rolling historical backtests.
        rerun_historical_forecasts = False
            Skips backtests.

    Regardless of rerun_historical_forecasts, this ALWAYS produces one
    forward-looking forecast fit on the full in-sample history, predicting
    into the rows where target_col is NaN (the 18-month extended window).
    """

    pdf = pdf.sort_values("Date").reset_index(drop=True)
    id_vals = {c: pdf[c].iloc[0] for c in ACT_GRP_COLS}

    driver_cols = [c for c in pdf.columns if c not in [*ACT_GRP_COLS, "Date", target_col]]
    drivers_used = len(driver_cols) > 0  #  computed once here, same as ARIMAX/SARIMAX

    in_sample = pdf[pdf[target_col].notna()].reset_index(drop=True)

    if len(in_sample) < MIN_TRAIN:  
        return pd.DataFrame()

    all_outputs = []

    # ============================================================
    # optional rolling historical backtests.
    # ============================================================
    if rerun_historical_forecasts:
        train_ends = range(MIN_TRAIN, len(in_sample) + 1, STEP_SIZE)

        for train_end in train_ends:
            train = in_sample.iloc[:train_end].copy()
            future = in_sample.iloc[train_end: train_end + FORECAST_HORIZON].copy()

            out = _prophet_fit_and_forecast(
                train, future, driver_cols, drivers_used, id_vals, driver_status
            )
            if out is None:
                continue
            all_outputs.append(out)
    # ============================================================
    # future forecast now ALWAYS runs, independent of rerun_historical_forecasts
    # ============================================================
    future_actuals = pdf[pdf[target_col].isna()].reset_index(drop=True).head(FORECAST_HORIZON)

    future_out = _prophet_fit_and_forecast(
        in_sample, future_actuals, driver_cols, drivers_used, id_vals, driver_status
    )
    if future_out is not None:
        all_outputs.append(future_out)
    # ============================================================

    if len(all_outputs) == 0:
        return pd.DataFrame()

    return pd.concat(all_outputs, ignore_index=True)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

if prophet_run:
    prophet_schema = StructType(
        [StructField(c, StringType(), False) for c in ACT_GRP_COLS] +
        [
            StructField("Training_End_Date", DateType(), False), 
            StructField("Forecast_Horizon", IntegerType(), False),
            StructField("Forecaster", StringType(), False),
            StructField("Drivers_Used_Flag", StringType(), False),
            StructField("Date", StringType(), False),
            StructField("Forecast", DoubleType(), True),
            StructField("Forecast_Lower", DoubleType(), True),
            StructField("Forecast_Upper", DoubleType(), True),
            StructField("best_changepoint_prior_scale", DoubleType(), True),
            StructField("best_seasonality_prior_scale", DoubleType(), True),
            StructField("cv_mae", DoubleType(), True),
        ]
    )


    prophet_output = actuals_fh_populated.groupBy(*ACT_GRP_COLS).applyInPandas(fit_prophet, schema=prophet_schema).cache()

    if rerun_historical_forecasts:
        prophet_output.write.mode('overwrite').parquet(Prophet_dir)
    else:
        prophet_output.write.mode('append').parquet(Prophet_dir)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### SARIMAX Forecasting

# CELL ********************

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from itertools import product
def _sarimax_fit_and_forecast(train, future, driver_cols, drivers_used, id_vals, driver_status):
    """
        Fits SARIMAX on 'train' and forecasts FORECAST_HORIZON steps
        to align with 'future'. returns an output DataFrame, or None, if 
        fitting failed / future window is short
    """

    if len(future) < FORECAST_HORIZON:
        return None
    
    ## Prepare target
    y = pd.to_numeric(train[target_col], errors="coerce")

    ## Prepare drivers
    exog_train = None
    exog_future = None
    if drivers_used:
        exog_train = train[driver_cols].apply(pd.to_numeric, errors='coerce')
        exog_future = future[driver_cols].apply(pd.to_numeric, errors='coerce')

        medians = exog_train.median()
        exog_train = exog_train.fillna(medians)
        future_medians = exog_future.median()
        exog_future = exog_future.fillna(future_medians)

    # Determining differencing order
    d = 0
    series_to_test = y.copy()
    while d <2:
        try:
            p_value = adfuller(series_to_test.dropna())[1]
        except Exception:
            break
        if p_value <0.05:
            break
        series_to_test = series_to_test.diff()
        d +=1
    
    # limits the differencing 
    if d>=1:
        D_options = [0]
    else:
        D_options = [0,1]

    # historical scale to limit unrealistic outputs
    hist_max_abs = float(y.abs().max())


    ## SARIMAX Grid (seasonal fits are more computationally expensive than linear ARIMAX)
    best_aic, best_order, best_model = np.inf, None, None
    for p, q in product(range(4), range(3)):
        for P, Q, D in product(range(2), range(2), D_options):

            try:
                m = sm.tsa.SARIMAX(
                    y, exog=exog_train, order=(p, d, q),
                    seasonal_order=(P,D,Q, SEASONAL_PERIODS),
                    enforce_stationarity=True, enforce_invertibility=True
                ).fit(disp=False)

                if not m.mle_retvals.get('converged', False):
                    continue

                ## check stability on AR/MA roots since converged isn't always reliable
                ## Roots too close to the unit circle indicate a near unit root / unstable model

                try:
                    ar_roots = m.polynomial_ar.roots() if hasattr(m.polynomial_ar, "roots") else np.roots(m.polynomial_ar)
                    if len(ar_roots) > 0 and np.any(np.abs(ar_roots) < 1.001):
                        continue
                except Exception:
                    pass

                if m.aic < best_aic:
                    ## sanity guard: reject any cadidate whose forecast magnitude is wqidly outside historical range
                    try:
                        trail_forecast = m.get_forecast(
                            steps=FORECAST_HORIZON, exog=exog_future
                        ).predicted_mean
                    except Exception:
                        continue
                    
                    if hist_max_abs > 0 and trail_forecast.abs().max() > 3 * hist_max_abs:
                        continue

                    best_aic = m.aic
                    best_order = (p,d,q,P,D,Q)
                    best_model = m
            except Exception:
                continue
    if best_model is None:
        return pd.DataFrame([])

    forecast_result = best_model.get_forecast(steps=FORECAST_HORIZON, exog=exog_future)
    forecast_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.05)

    out = pd.DataFrame({
        **{c: id_vals[c] for c in ACT_GRP_COLS},
        "Training_End_Date": train["Date"].iloc[-1],
        "Forecast_Horizon": range(1, FORECAST_HORIZON +1),
        "Forecaster": "SARIMAX",
        "Drivers_Used_Flag": driver_status,
        "Date": future['Date'].astype(str).values,
        "Forecast": forecast_mean.values,
        "Forecast_Lower": conf_int.iloc[:,0].values,
        "Forecast_Upper": conf_int.iloc[:,1].values,
        "best_order": str(best_order),
        "aic": best_aic,
    })

    return out

def fit_sarimax(pdf):
    """
        Fits SARIM or SARIMAX depending on whether driver variables exist
        Behavior:
            rerun historical forecasts = true
                additionally preforms rolling historical backtests
            false:
                skips back tests
        regardless of historical run setting this will always produce one forward
        looking forecast fit on the full in sample history, predicting into the rows where 
        target_col is NaN (the 18 month extended window based on Forecast Horizon)
    """
    # Sort observations
    pdf = pdf.sort_values('Date').reset_index(drop=True)

    id_vals = {c: pdf[c].iloc[0] for c in ACT_GRP_COLS}

    # Determine driver cols
    driver_cols = [c for c in pdf.columns if c not in [*ACT_GRP_COLS, 'Date', target_col]]
    drivers_used = len(driver_cols)>0

    # Historical observations (non null target)
    in_sample = pdf[pdf[target_col].notna()].reset_index(drop=True)

    if len(in_sample) < MIN_TRAIN:
        return pd.DataFrame()
    
    all_outputs =  []

    ## HISTORICAL RERUNS / BACK TESTS
    if rerun_historical_forecasts:
        train_ends = range(MIN_TRAIN, len(in_sample)+1, STEP_SIZE)

        for train_end in train_ends:
            train = in_sample.iloc[:train_end].copy()
            future = in_sample.iloc[train_end: train_end + FORECAST_HORIZON].copy()

            out = _sarimax_fit_and_forecast(
                train, future, driver_cols, drivers_used, id_vals, driver_status
            )

            if out is None:
                continue
            all_outputs.append(out)
    
    ## FUTURE LOOKING FORECASTING
    future_actuals = pdf[pdf[target_col].isna()].reset_index(drop=True).head(FORECAST_HORIZON)

    future_out = _sarimax_fit_and_forecast(
        in_sample, future_actuals, driver_cols, drivers_used, id_vals, driver_status
    )

    if future_out is not None:
        all_outputs.append(future_out)

    # return results
    if len(all_outputs)==0:
        return pd.DataFrame()


    return pd.concat(all_outputs, ignore_index=True)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

if sarimax_run:
    # ==========================================================
    # Output Schema
    # ==========================================================

    sarimax_schema = StructType(
        [StructField(c, StringType(), False) for c in ACT_GRP_COLS] +
        [
            StructField("Training_End_Date", DateType(), False),
            StructField("Forecast_Horizon", IntegerType(), False),
            StructField("Forecaster", StringType(), False),
            StructField("Drivers_Used_Flag", StringType(), False),
            StructField("Date", StringType(), False),
            StructField("Forecast", DoubleType(), True),
            StructField("Forecast_Lower", DoubleType(), True),
            StructField("Forecast_Upper", DoubleType(), True),
            StructField("best_order", StringType(), True),
            StructField("aic", DoubleType(), True),
        ]
    )



    sarimax_output = actuals_fh_populated.groupBy(*ACT_GRP_COLS).applyInPandas(fit_sarimax, schema=sarimax_schema).cache()

    if rerun_historical_forecasts:
        sarimax_output.write.mode('overwrite').parquet(Sarimax_dir)
    else:
        sarimax_output.write.mode('append').parquet(Sarimax_dir)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### ARIMAX Forecasting

# CELL ********************

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from itertools import product


# ==========================================================
# ARIMA / ARIMAX Model
# ==========================================================

def _arimax_fit_and_forecast(train, future, driver_cols, drivers_used, id_vals, driver_status):
    """
    Fits ARIMA/ARIMAX on `train` and forecasts FORECAST_HORIZON steps
    to align with `future`. Returns an output DataFrame, or None if
    fitting failed / future window is short.
    """
    if len(future) < FORECAST_HORIZON:
        return None

    # Prepare target
    y = pd.to_numeric(train[target_col], errors="coerce")

    # Prepare drivers
    exog_train = None
    exog_future = None
    if drivers_used:
        exog_train = train[driver_cols].apply(pd.to_numeric, errors="coerce")
        exog_future = future[driver_cols].apply(pd.to_numeric, errors="coerce")

        medians = exog_train.median()
        exog_train = exog_train.fillna(medians)
        future_medians = exog_future.median()
        exog_future = exog_future.fillna(future_medians)

    # --------------------------------------------------
    # Determine differencing order
    # --------------------------------------------------
    d = 0
    series = y.copy()
    while d < 2:
        try:
            p_value = adfuller(series.dropna())[1]
        except Exception:
            break
        if p_value < 0.05:
            break
        series = series.diff()
        d += 1

    # historical scale
    hist_max_abs = float(y.abs().max())

    # --------------------------------------------------
    # Grid Search
    # --------------------------------------------------
    best_model = None
    best_order = None
    best_aic = np.inf

    for p, q in product(range(3), range(3)):  # solution matrix of (p, q) combos
        try:
            model_kwargs = {"order": (p, d, q)}
            if drivers_used:
                model_kwargs["exog"] = exog_train

            model = ARIMA(y, **model_kwargs).fit()

            # skip models that failed to converge
            if not model.mle_retvals.get('converged', False):
                continue

            ## reject unstable AR and MA models
            try:
                if len(model.arroots) > 0:
                    if np.any(np.abs(model.arroots) <= 1):
                        continue

                if len(model.maroots) > 0:
                    if np.any(np.abs(model.maroots) <= 1):
                        continue

            except Exception:
                continue
            

            if model.aic < best_aic:
                try:
                    trail_kwargs = {'steps': FORECAST_HORIZON}
                    if drivers_used:
                        trail_kwargs['exog'] = exog_future
                    trail_forecast = model.get_forecast(**trail_kwargs).predicted_mean
                except Exception:
                    continue
                
                if hist_max_abs > 0 and trail_forecast.abs().max() > 3 * hist_max_abs:
                    continue
                best_aic = model.aic
                best_order = (p, d, q)
                best_model = model

        except Exception:
            continue

    if best_model is None:
        return None

    # --------------------------------------------------
    # Forecast
    # --------------------------------------------------
    forecast_kwargs = {"steps": FORECAST_HORIZON}
    if drivers_used:
        forecast_kwargs["exog"] = exog_future

    forecast_result = best_model.get_forecast(**forecast_kwargs)
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    out = pd.DataFrame({
        **{c: id_vals[c] for c in ACT_GRP_COLS},
        "Training_End_Date": train["Date"].iloc[-1],
        "Forecast_Horizon": range(1, FORECAST_HORIZON + 1),
        "Forecaster": "ARIMAX",
        "Drivers_Used_Flag": driver_status,
        "Date": future["Date"].astype(str).values,
        "Forecast": forecast.values,
        "Forecast_Lower": conf_int.iloc[:, 0].values,
        "Forecast_Upper": conf_int.iloc[:, 1].values,
        "best_order": str(best_order),
        "aic": best_aic,
    })


    return out


def fit_arimax(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Fits either ARIMA or ARIMAX depending on whether driver variables exist.

    Behavior:
        rerun_historical_forecasts = True
            Additionally performs rolling historical backtests.
        rerun_historical_forecasts = False
            Skips backtests.

    Regardless of rerun_historical_forecasts, this ALWAYS produces one
    forward-looking forecast fit on the full in-sample history, predicting
    into the rows where target_col is NaN (e.g. your 18-month extended
    actuals window used to carry forward external drivers).
    """
    # Sort observations
    pdf = pdf.sort_values("Date").reset_index(drop=True)

    id_vals = {c: pdf[c].iloc[0] for c in ACT_GRP_COLS}

    # Determine driver columns
    driver_cols = [c for c in pdf.columns if c not in [*ACT_GRP_COLS, "Date", target_col]]
    drivers_used = len(driver_cols) > 0

    # Historical observations (non-null target)
    in_sample = pdf[pdf[target_col].notna()].reset_index(drop=True)

    if len(in_sample) < MIN_TRAIN:
        return pd.DataFrame()

    all_outputs = []

    # --------------------------------------------------
    # Optional rolling historical backtests
    # --------------------------------------------------
    if rerun_historical_forecasts:
        train_ends = range(MIN_TRAIN, len(in_sample) + 1, STEP_SIZE)

        for train_end in train_ends:
            train = in_sample.iloc[:train_end].copy()
            future = in_sample.iloc[train_end: train_end + FORECAST_HORIZON].copy()

            out = _arimax_fit_and_forecast(
                train, future, driver_cols, drivers_used, id_vals, driver_status
            )
            if out is None:
                continue

            all_outputs.append(out)

    # --------------------------------------------------
    # ALWAYS run the true forward-looking forecast
    # (fit on full in-sample history, forecast into the
    # NaN-target rows from the 18-month extended window)
    # --------------------------------------------------
    future_actuals = pdf[pdf[target_col].isna()].reset_index(drop=True).head(FORECAST_HORIZON)

    future_out = _arimax_fit_and_forecast(
        in_sample, future_actuals, driver_cols, drivers_used, id_vals, driver_status
    )

    if future_out is not None:
        all_outputs.append(future_out)

    # ------------------------------------------------------
    # Return Results
    # ------------------------------------------------------
    if len(all_outputs) == 0:
        return pd.DataFrame()

    return pd.concat(all_outputs, ignore_index=True)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

if arimax_run:
    # ==========================================================
    # Output Schema
    # ==========================================================

    arimax_schema = StructType(
        [StructField(c, StringType(), False) for c in ACT_GRP_COLS] +
        [
            StructField("Training_End_Date", DateType(), False),
            StructField("Forecast_Horizon", IntegerType(), False),
            StructField("Forecaster", StringType(), False),
            StructField("Drivers_Used_Flag", StringType(), False),
            StructField("Date", StringType(), False),
            StructField("Forecast", DoubleType(), True),
            StructField("Forecast_Lower", DoubleType(), True),
            StructField("Forecast_Upper", DoubleType(), True),
            StructField("best_order", StringType(), True),
            StructField("aic", DoubleType(), True),
        ]
    )



    arimax_output = actuals_fh_populated.groupBy(*ACT_GRP_COLS).applyInPandas(fit_arimax, schema=arimax_schema).cache()

    if rerun_historical_forecasts:
        arimax_output.write.mode('overwrite').parquet(Arimax_dir)
    else:
        arimax_output.write.mode('append').parquet(Arimax_dir)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Holts ETS Forecasting

# CELL ********************

def fit_ets(pdf):


    ## Creation of an empty dataframe to return if required for edge cases
    EMPTY_DF = pd.DataFrame({
        c: pd.Series(dtype="object") for c in ACT_GRP_COLS
    } | {
        "Training_End_Date": pd.Series(dtype="datetime64[ns]"),
        "Forecast_Horizon": pd.Series(dtype="int32"),
        "Forecaster": pd.Series(dtype="object"),
        "Drivers_Used_Flag": pd.Series(dtype="int32"),
        "Date": pd.Series(dtype="object"),
        "Forecast": pd.Series(dtype="float64"),
        "best_trend": pd.Series(dtype="object"),
        "best_seasonal": pd.Series(dtype="object"),
        "best_damped": pd.Series(dtype="int32"),
        "aic": pd.Series(dtype="float64")
    })


    pdf = pdf.sort_values('Date').reset_index(drop=True)
    id_vals = {c: pdf[c].iloc[0] for c in ACT_GRP_COLS}

    y = pd.to_numeric(pdf[target_col], errors='coerce')

    if y.isna().any():
        y = y.fillna(y.median()) ## ETS requires a complete series: median fill gaps

    if len(y) < 24:
        return EMPTY_DF.copy()

    # ETS requires strictly positive values for multiplicative trend/seasonal —
    # check before including "mul" options in the grid
    allow_mul = (y > 0).all()

    grid = []
    for trend in [None,"add", "mul"] if allow_mul else [None,"add"]:
        for seasonal in [None,"add", "mul"] if allow_mul else [None,"add"]:
            for damped in [True, False]:
                if trend is None and damped:
                    continue
                grid.append({"trend": trend, "seasonal": seasonal, "damped_trend": damped})

    #################################################
    ## Completing back testing iteration or only future forecasting
    #################################################
    all_outputs = []


    if rerun_historical_forecasts:
        train_ends = range(MIN_TRAIN, len(y)+1, STEP_SIZE)
    else:
        # only fit once using the full history
        train_ends = [len(y)]

    ###############################################
    ## FIT MODELS
    ###############################################

    for train_end in train_ends:
    
        best_aic, best_params, best_model = np.inf, None, None

        train = y.iloc[:train_end]

        for params in grid:
            try:
                m = ExponentialSmoothing(
                    train, trend=params["trend"], seasonal=params["seasonal"],
                    seasonal_periods=SEASONAL_PERIODS, damped_trend=params["damped_trend"],
                    initialization_method="estimated"
                ).fit(optimized=True)
                if m.aic < best_aic:
                    best_aic, best_params, best_model = m.aic, params, m
            except Exception as e:
                last_exception = e
                continue

        if best_model is None:
            print(id_vals, last_exception)
            return EMPTY_DF.copy()

        forecast = best_model.forecast(FORECAST_HORIZON)
        forecast = forecast.replace([np.inf, -np.inf], np.nan)
        last_train_date = pd.to_datetime(pdf['Date'].iloc[train_end-1])
        future_dates = pd.date_range(
            last_train_date,
            periods=FORECAST_HORIZON+1,
            freq='MS'
        )[1:]


        out = pd.DataFrame({
            **{c: id_vals[c] for c in ACT_GRP_COLS},
            "Forecast_Horizon": range(1, FORECAST_HORIZON + 1),
            "Training_End_Date": pdf['Date'].iloc[train_end-1],
            "Forecaster": "HOLTS_ETS",
            "Drivers_Used_Flag": int(drivers_used),
            "Date": future_dates.astype(str),
            "Forecast": forecast.values,
            "best_trend": best_params["trend"],
            "best_seasonal": best_params["seasonal"],
            "best_damped": int(best_params["damped_trend"]),
            "aic": best_aic,
        })

        all_outputs.append(out)


    if len(all_outputs)==0:
        return EMPTY_DF.copy()

    return pd.concat(all_outputs, ignore_index=True)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

if ets_run:
    ets_schema = StructType(
        [StructField(c, StringType(), False) for c in ACT_GRP_COLS] +
        [
            StructField("Training_End_Date", DateType(), False),
            StructField("Forecast_Horizon", IntegerType(), False),
            StructField("Forecaster", StringType(), False),
            StructField("Drivers_Used_Flag", IntegerType(), False),
            StructField("Date", StringType(), False),
            StructField("Forecast", DoubleType(), True),
            StructField("best_trend", StringType(), True),
            StructField("best_seasonal", StringType(), True),
            StructField("best_damped", IntegerType(), True),
            StructField("aic", DoubleType(), True)
        ]
    )

    ets_forecasts = (
        actuals
        .fillna({target_col:0})
        .groupBy(*ACT_GRP_COLS)
        .applyInPandas(fit_ets, schema=ets_schema)
    ).cache()


    if rerun_historical_forecasts:
        ets_forecasts.write.mode('overwrite').parquet(Holts_dir)
    else:
        ets_forecasts.write.mode('append').parquet(Holts_dir)

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
