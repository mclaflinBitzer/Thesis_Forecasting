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

## BASE DIRECTORY FOR ALL OUTPUTS
excel_base_dir = "/lakehouse/default/Files/Automated_Driver_Analysis/"


## TOPLINE
T_series = ['series']
T_initial_target_col = "Quantity"

T_DRV_GRP_COLS = ['Indicator']
T_ACT_GRP_COLS = ['Product_Category','series']

T_cols = list(dict.fromkeys(T_ACT_GRP_COLS + T_DRV_GRP_COLS))

T_ACT_COLS_RENAME = {"Date":"target_date","residual":"target_residual"}  
T_DRV_COLS_RENAME = {"Date":"feature_date","residual":"feature_residual"}


T_actuals_table = "Sales_Forecasting.silver.topline_cutoff_data"

T_feature_diagnostics_file = excel_base_dir + "Topline/topline_ENCV_feature_diagnostics.xlsx"
T_selected_feature_file = excel_base_dir + "Topline/topline_ENCV_selected_features.xlsx"

T_xgboost_diagnostics_file = excel_base_dir + "Topline/topline_xgboost_feature_diagnostics.xlsx"
T_xgboost_selected_feature_file = excel_base_dir + "Topline/topline_xgboost_selected_feature.xlsx"

T_dl_selected_feature_file = excel_base_dir + "Topline/topline_dl_selected_feature.xlsx"


## MIDDLE
M_series = ['series']
M_initial_target_col = 'Quantity'

M_DRV_GRP_COLS = ['Region','Indicator']
M_ACT_GRP_COLS = ['Product_Category', 'Region','series']

M_cols = list(dict.fromkeys(M_ACT_GRP_COLS + M_DRV_GRP_COLS))

M_ACT_COLS_RENAME = {"Date":"target_date","residual":"target_residual"}  
M_DRV_COLS_RENAME = {"Date":"feature_date","residual":"feature_residual"}


M_actuals_table = "Sales_Forecasting.silver.middle_cutoff_data"
M_feature_diagnostics_file = excel_base_dir + "Middle/middle_feature_diagnostics.xlsx"
M_selected_feature_file = excel_base_dir + "Middle/middle_selected_features.xlsx"

M_xgboost_diagnostics_file = excel_base_dir + "Middle/middle_xgboost_feature_diagnostics.xlsx"
M_xgboost_selected_feature_file = excel_base_dir + "Middle/middle_xgboost_selected_features.xlsx"

M_dl_selected_feature_file = excel_base_dir + "middle_dl_selected_feature.xlsx"


## shared
col_renamed = {"Quantity":"target_value","Value":"feature_value"}
driver_table = "Sales_Forecasting.silver.compiled_drivers"



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## BASELINE output directories
parquet_dir = "abfss://991f5e4b-c174-4ff2-992e-feb17d49d25a@onelake.dfs.fabric.microsoft.com/22746de3-183e-4327-a844-dceda0b7165c/Files/Forecasting"

Topline = False
drivers_used = False
rerun_historical_forecasts = True
ets_forecast = True


if Topline:

    actuals_table = "Sales_Forecasting.silver.topline_cutoff_data"
    initial_target_col = "Quantity"

    DRV_GRP_COLS = ['Indicator']
    ACT_GRP_COLS = ['Product_Category','series']

    ## OUTPUT DIRECTORIES
    Holts_dir = parquet_dir + "/Topline/Holts_Output.parquet"

else:

    actuals_table = "Sales_Forecasting.silver.middle_cutoff_data"
    initial_target_col = 'Quantity'

    DRV_GRP_COLS = ['Region','Indicator']
    ACT_GRP_COLS = ['Product_Category', 'Region','series']

    ## OUTPUT DIRECTORIES
    Holts_dir = parquet_dir + "/Middle/Holts_Output.parquet"


# SHARED PARAMETERS

## Model Parameters
FORECAST_HORIZON = 18
SEASONAL_PERIODS = 12
MIN_TRAIN = 24 # minimum history of data before fitting (ETS model)
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

actuals = spark.read.table(actuals_table).withColumnRenamed(initial_target_col, target_col)

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



def fit_arimax(pdf):
    """
    pdf must contain: Date, target_col, and the SELECTED driver columns already
    pivoted wide, PLUS future rows (Date > last actual date) with driver values
    populated (from your pre-forecasted driver sources) and target_col = NaN.
    This lets exog be sliced cleanly into in-sample vs. forecast-period.
    """

    pdf = pdf.sort_values("Date").reset_index(drop=True)
    id_vals = {c: pdf[c].iloc[0] for c in ACT_GRP_COLS}

    driver_cols = [c for c in pdf.columns if c not in [*ACT_GRP_COLS, "Date", target_col]]

    in_sample = pdf[pdf[target_col].notna()].reset_index(drop=True)
    future = pdf[pdf[target_col].isna()].reset_index(drop=True)

    # Historical rerun vs not based on parameter status
    if rerun_historical_forecasts:
        train_ends = range(MIN_TRAIN, len(in_sample)+1, STEP_SIZE)
    else:
        train_ends = [len(in_sample)]

    for train_end in train_ends:
        train_y = y.iloc[:train_end]

        future_exog = exog_train.iloc[train_end: train_end+FORECAST_HORIZON]

        forecast_dates = (in_sample['Date'].iloc[train_end:train_end+FORECAST_HORIZON])
        



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

arimax_schema = StructType(
    [StructField(c, StringType(), False) for c in ACT_GRP_COLS] +
    [
        StructField("Training_End_Date", DateType(), False),
        StructField("Forecast_Horizon", IntegerType(), False),
        StructField("Forecaster", StringType(), False),
        StructField("Forecast", DoubleType(), True),
        StructField("Forecast_Lower", DoubleType(), True),
        StructField("Forecast_Upper", DoubleType(), True),
        StructField("best_order", StringType(), True),
        StructField("aic", DoubleType(), True)
    ]
)



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

if ets_forecasts:
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
        #.filter(col('series')=="SCREWS___APAC")
        .filter(col(target_col).isNotNull())
        .groupBy(*ACT_GRP_COLS)
        .applyInPandas(fit_ets, schema=ets_schema)
    )


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

display(ets_forecasts)

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
