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

# MARKDOWN ********************

# ### Stage 1: Driver Classification
# 
# **Purpose:** Select correct transformations
# 
# #### Trending Variables
# **Examples:** GDP, production, commodities  
# **Traits:** trend, non-stationary  
# **Transforms:** YoY, growth rates, differencing  
# 
# #### Non-Trending Variables
# **Examples:** inflation, interest rates, utilization  
# **Traits:** stable mean, mean-reverting  
# **Transforms:** usually none (check stationarity)
# 
# ---
# 
# ### Stage 2: Feature Engineering
# 
# **Purpose:** Create candidate predictors
# 
# **Features:**
# - Value (t)
# - Lags: 1-24
# - Leads: 1-24
# - Rolling: mean / median / std (3/6/12M)
# - Growth: MoM, QoQ, YoY
# 
# ---
# 
# ### Stage 3: Target Transformation
# 
# **Purpose:** Align target + drivers
# 
# **If trending:**
# - YoY growth
# - Differencing
# - Detrending
# 
# **If stationary:**
# - Use levels
# 
# ---
# 
# ### Stage 4: Cross-Correlation
# 
# **Purpose:** Identify lead/lag effects
# 
# **Evaluate:**
# - Target vs driver level
# - Target vs lags
# - Target vs leads
# 
# **Metrics:**
# - Pearson correlation (linear)
# - Cross-correlation function (lead/lag)
# 
# **Rule:**
# Prefer stable lag ranges, not single spikes
# 
# ---
# 
# ### Stage 5: Correlation Filtering
# 
# **Purpose:** Remove weak signals
# 
# - Keep |corr| > 0.20 or top N
# - Check stability across windows
# - Ensure sufficient observations
# 
# ---
# 
# ### Stage 6: Multicollinearity
# 
# **Purpose:** Remove redundant drivers
# 
# - Drop |corr| > 0.80
# - VIF > 5–10
# 
# ---
# 
# ### Stage 7: Economic Validation
# 
# **Purpose:** Ensure business logic
# 
# **Checks:**
# - Causality plausible
# - Lag realistic
# - Interpretable relationship
# 
# **Example:**
# ✔ Industrial production → demand  
# ✖ Random unrelated signal
# 
# ---
# 
# ### Stage 8: Final Selection
# 
# **Scoring weights:**
# - Correlation: 30%
# - Stability: 20%
# - Forecast availability: 20%
# - Economic logic: 15%
# - Model importance: 15%
# 
# **Importance methods:**
# - Random Forest
# - XGBoost
# - SHAP
# - Permutation importance


# CELL ********************

from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.functions import col
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.types import *
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from pyspark.sql.utils import AnalysisException
import numpy as np
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from statsmodels.tsa.stattools import kpss
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from matplotlib.ticker import PercentFormatter
from pyspark.sql.functions import pandas_udf

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

compiled_drivers = spark.read.table("Sales_Forecasting.silver.compiled_drivers")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Stage 1: Driver Classification
# 
# **Purpose:** Select correct transformations
# 
# Decide how each driver should be transformed.
# Implementation
# For each driver:
# 
# Step 1: Stationarity test
# 
# • ADF test or KPSS
# • Visual check (rolling mean/variance)
# 
# Step 2: Classify
# 
# - If strong trend → "Trending"
# - If mean-reverting → "Non-trending"
# 
# Step 3: Store metadata
# Create a driver config table:
# 
# | Driver | Type | Transformation
# 
# 
# 
# #### Trending Variables
# **Examples:** GDP, production, commodities  
# **Traits:** trend, non-stationary  
# **Transforms:** YoY, growth rates, differencing  
# 
# #### Non-Trending Variables
# **Examples:** inflation, interest rates, utilization  
# **Traits:** stable mean, mean-reverting  
# **Transforms:** usually none (check stationarity)


# MARKDOWN ********************

# #### ADF test

# CELL ********************

def adf_group(df):
    pdf = df.sort_values("Date")

    country = pdf["Country"].iloc[0]
    indicator = pdf["Indicator"].iloc[0]

    ts = (
        pd.to_numeric(pdf["Value"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .values
    )


    if len(ts) < 12:
        return pd.DataFrame([{
            "Country": pdf["Country"].iloc[0],
            "Indicator": pdf["Indicator"].iloc[0],
            "adf_stat": None,
            "adf_p_value": None,
            "adf_stationary_flag": "Insufficient Data"            
        }])
    
    if np.nanstd(ts) == 0:
        return pd.DataFrame([{
            "Country": country,
            "Indicator": indicator,
            "adf_stat": None,
            "adf_p_value": None,
            "adf_stationary_flag": "Constant Series (Skipped)"
        }])


    ## ADF execution
    try:
        adf_stat, p_value, *_ = adfuller(ts)

        return pd.DataFrame([{
            "Country": country,
            "Indicator": indicator,
            "adf_stat": adf_stat,
            "adf_p_value": p_value,
            "adf_stationary_flag": (
                "Strongly Stationary" if p_value < 0.01
                else "Stationary" if p_value < 0.05
                else "Weakly Non Stationary" if p_value < 0.1
                else "Non Stationary"
            )
        }])

    except Exception as e:
        return pd.DataFrame([{
            "Country": country,
            "Indicator": indicator,
            "adf_stat": None,
            "adf_p_value": None,
            "adf_stationary_flag": f"ADF Failed: {str(e)}"
        }])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

adf_schema = StructType([
    StructField("Country", StringType(), True),
    StructField("Indicator", StringType(), True),
    StructField("adf_stat", DoubleType(), True),
    StructField("adf_p_value", DoubleType(), True),
    StructField("adf_stationary_flag", StringType(), True)
])

adf_results = (compiled_drivers
                    .groupBy("Country","Indicator")
                    .applyInPandas(adf_group, schema=adf_schema)
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### KPSS Test

# CELL ********************

def kpss_group(df):
    pdf = df.sort_values("Date")

    country = pdf["Country"].iloc[0]
    indicator = pdf["Indicator"].iloc[0]

    # FORCE CLEAN NUMERIC PIPELINE
    ts = (
        pd.to_numeric(pdf["Value"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .values
    )


    if len(ts) < 12:
        return pd.DataFrame([{
            "Country": country,
            "Indicator": indicator,
            "kpss_stat": None,
            "kpss_p_value": None,
            "kpss_stationary_flag": "Insufficient Data (<12 obs)"
        }])
    
    # SAFETY CHECK 2: constant series
    if np.nanstd(ts) == 0:
        return pd.DataFrame([{
            "Country": country,
            "Indicator": indicator,
            "kpss_stat": None,
            "kpss_p_value": None,
            "kpss_stationary_flag": "Constant Series (Skipped)"
        }])

        
    ## Run KPSS
    try:
        kpss_stat, p_value, _, _ = kpss(
            ts,
            regression="c",
            nlags="auto"
        )

        return pd.DataFrame([{
                "Country": country,
                "Indicator": indicator,
                "kpss_stat": kpss_stat,
                "kpss_p_value": p_value,
                "kpss_stationary_flag": 
                    "Strongly Stationary" if p_value >= 0.10
                    else "Stationary" if p_value >= 0.05
                    else "Weakly Non-Stationary" if p_value >= 0.01
                    else "Non-Stationary"
        }])


    except Exception as e:
        return pd.DataFrame([{
                "Country": country,
                "Indicator": indicator,
                "kpss_stat": None,
                "kpss_p_value": None,
                "kpss_stationary_flag": f"KPSS Failed: {str(e)}"
        }])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

kpss_schema = StructType([
                    StructField("Country", StringType(), False),
                    StructField("Indicator", StringType(), False),
                    StructField("kpss_stat", DoubleType(), True),
                    StructField("kpss_p_value", DoubleType(), True),
                    StructField("kpss_stationary_flag", StringType(), False)
])

kpss_results = (compiled_drivers.groupBy("Country", "Indicator")
                    .applyInPandas(kpss_group, schema=kpss_schema))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### compilation of adf & kpss outputs

# CELL ********************

df_stationary_stats = adf_results.join(kpss_results, ["Country", "Indicator"])


driver_final_stats = df_stationary_stats.withColumn(
    "ADF_status",
    when(col("adf_stationary_flag").contains("Constant"), "Constant")
    .when(col("adf_stationary_flag").contains("Failed"), "Failed")
    .otherwise("Valid")
)

driver_final_stats = driver_final_stats.withColumn(
    "KPSS_status",
    when(col("kpss_stationary_flag").contains("Constant"), "Constant")
    .when(col("kpss_stationary_flag").contains("Failed"), "Failed")
    .otherwise("Valid")
)


driver_final_stats = driver_final_stats.withColumn(
    "ADF_is_stationary",
    when(col("ADF_status") != "Valid", None)
    .when(col("adf_p_value") < 0.05, 1)
    .otherwise(0)
)

driver_final_stats = driver_final_stats.withColumn(
    "KPSS_is_stationary",
    when(col("KPSS_status") != "Valid", None)
    .when(col("kpss_p_value") > 0.05, 1)
    .otherwise(0)
)

driver_final_stats = driver_final_stats.withColumn(
    "stationarity_score",
    coalesce(col("ADF_is_stationary"), lit(0)) +
    coalesce(col("KPSS_is_stationary"), lit(0))
)


driver_final_stats = driver_final_stats.withColumn(
        "Final_Stationarity_Flag",
        when(
            (col("ADF_status") == "Constant") | (col("KPSS_status") == "Constant"),
            "Constant Series"
        )
        .when(
            (col("ADF_status") == "Failed") | (col("KPSS_status") == "Failed"),
            "Test Failed / Unreliable"
        )
        .when(col("stationarity_score") == 2, "Strongly Stationary")
        .when(col("stationarity_score") == 1, "Weakly Stationary / Unstable")
        .otherwise("Non-Stationary")
    )\
    .drop("ADF_status","KPSS_status","ADF_is_stationary","KPSS_is_stationary","stationarity_score")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

driver_final_stats.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration.driver_stationary_stats")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Stage 2: Feature Engineering
# **Purpose:** Create candidate predictors
# **Features:**
# - Value (t)
# - Lags: 1-24
# - Leads: 1-24
# - Rolling: mean / median / std (3/6/12M)
# - Growth: MoM, QoQ, YoY
# ---

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

# MARKDOWN ********************

# ### Stage 3: Target Transformation
# **Purpose:** Align target + drivers
# **If trending:**
# - YoY growth
# - Differencing
# - Detrending
# **If stationary:**
# - Use levels
# ---

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Stage 4: Cross-Correlation
# **Purpose:** Identify lead/lag effects
# **Evaluate:**
# - Target vs driver level
# - Target vs lags
# - Target vs leads
# **Metrics:**
# - Pearson correlation (linear)
# - Cross-correlation function (lead/lag)
# **Rule:**
# Prefer stable lag ranges, not single spikes
# ---

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

# MARKDOWN ********************

# 
# ### Stage 5: Correlation Filtering
# **Purpose:** Remove weak signals
# - Keep |corr| > 0.20 or top N
# - Check stability across windows
# - Ensure sufficient observations
# ---

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Stage 6: Multicollinearity
# **Purpose:** Remove redundant drivers
# - Drop |corr| > 0.80
# - VIF > 5–10
# ---

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

# MARKDOWN ********************

# ### Stage 7: Economic Validation
# **Purpose:** Ensure business logic
# **Checks:**
# - Causality plausible
# - Lag realistic
# - Interpretable relationship
# **Example:**
# ✔ Industrial production → demand  
# ✖ Random unrelated signal
# ---

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

# MARKDOWN ********************

# ### Stage 8: Final Selection
# **Scoring weights:**
# - Correlation: 30%
# - Stability: 20%
# - Forecast availability: 20%
# - Economic logic: 15%
# - Model importance: 15%
# **Importance methods:**
# - Random Forest
# - XGBoost
# - SHAP
# - Permutation importance

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
