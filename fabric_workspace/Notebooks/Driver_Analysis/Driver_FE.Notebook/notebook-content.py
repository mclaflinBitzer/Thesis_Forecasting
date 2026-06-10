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

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Welcome to your new notebook
# Type here in the cell editor to add code!
compiled_drivers = spark.read.table("Sales_Forecasting.silver.compiled_drivers")

display(compiled_drivers.limit(100))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import *

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(compiled_drivers.filter(~col("Indicator").like("%Real USD%")).select("Indicator", "Value").distinct())

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

def adf_test(df):
    results = []

    series = df.select("Country","Indicator").distinct().collect()

    for row in series:

        country = row["Country"]
        indicator = row["Indicator"]


        serie_df = (
            df
            .filter((col("Country")==country) & (col("Indicator")==indicator))
            .orderBy("Date")
            .toPandas()  
        )

        ts = (
            serie_df["Value"]
            .astype(float)
            .dropna()
            .values
        )

        # minimum sample size guard
        if len(ts) < 12:
            results.append({
                "Country": country,
                "Indicator": indicator,
                "adf_stat": None,
                "adf_p_value": None,
                "adf_stationary_flag": "Insufficient Data (<12 obs)"
            })

            continue
        
        ## ADF execution
        try:
            adf_stat, p_value, *_ = adfuller(ts)

            results.append({
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
            })

        except Exception as e:
            results.append({
                "Country": country,
                "Indicator": indicator,
                "adf_stat": None,
                "adf_p_value": None,
                "adf_stationary_flag": f"ADF Failed: {str(e)}"
            })

    ## Convert to Spark DF
    adf_schema = StructType([
        StructField("Country", StringType(), False),
        StructField("Indicator", StringType(), False),
        StructField("adf_stat", DoubleType(), True),
        StructField("adf_p_value", DoubleType(), True),
        StructField("adf_stationary_flag", StringType(), False)
    ])
    adf_results = spark.createDataFrame( results, schema=adf_schema)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

drivers_adf = adf_test(compiled_drivers)

display(drivers_adf)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(compiled_drivers.filter(col("Country")=="World").orderBy("Indicator","Date"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### KPSS Test

# CELL ********************

def kpss_test(df):
    results = []

    series = df.select("Country","Indicator").distinct().collect()

    for row in series:

        country = row["Country"]
        indicator = row["Indicator"]

        serie_df = (
            df
            .filter((col("Country")==country) & (col("Indicator")==indicator))
            .orderBy("Date")
            .toPandas()
        )

        ts = (
            series_df["Value"]
            .astype(float)
            .dropna()
            .values
        )

        ## minimum sample size guardrail
        if len(ts) < 12:
            results.append({
                "Country": country,
                "Indicator": indicator,
                "kpss_stat": None,
                "kpss_p_value": None,
                "kpss_stationary_flag": "Insufficient Data (<12 obs)"
            })

            continue
        
        ## Run KPSS
        try:
            kpss_stat, p_value, _, _ = kpss(
                ts,
                regression="c",
                nlags="auto"
            )

            results.append({
                "Country": country,
                "Indicator": indicator,
                "kpss_stat": kpss_stat,
                "kpss_p_value": p_value,
                "kpss_stationary_flag": 
                    "Strongly Stationary" if p_value >= 0.10
                    else "Stationary" if p_value >= 0.05
                    else "Weakly Non-Stationary" if p_value >= 0.01
                    else "Non-Stationary"
            })
        
        except Exception as e:
            results.append([
                "Country": country,
                "Indicator": indicator,
                "kpss_stat": None,
                "kpss_p_value": None,
                "kpss_stationary_flag": f"KPSS Failed: {str(e)}"               
            ])

    ## Create Spark Dataframe
    kpss_schema = StructType([
        StructField("Country", StringType(), False),
        StructField("Indicator", StringType(), False),
        StructField("kpss_stat", DoubleType(), True),
        StructField("kpss_p_value", DoubleType(), True),
        StructField("kpss_stationary_flag", StringType(), False)
    ])

    kpss_df = spark.createDataFrame(results, schema=kpss_schema)

    return kpss_df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

driver_kpss = kpss_test(compiled_drivers)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

driver_stationary_stats = drivers_adf.join(driver_kpss, ["Country", "Indicator"])

driver_final_stats = (
    df_stationary_stats
    .withColumn("ADF_is_stationary", when(col("adf_p_value") < 0.05, 1).otherwise(0))
    .withColumn("KPSS_is_stationary", when(col("kpss_p_value") > 0.05, 1).otherwise(0))
)


# -----------------------------
# Composite score
# -----------------------------
driver_final_stats = driver_final_stats.withColumn(
    "stationarity_score",
    col("ADF_is_stationary") +
    col("KPSS_is_stationary")
)

# -----------------------------
# Final stationarity flag (base logic + conflict handling)
# -----------------------------
driver_final_stats = driver_final_stats.withColumn(
            "Final_Stationarity_Flag",
            when(
                (col("ADF_is_stationary") == 1) &
                (col("KPSS_is_stationary") == 0),
                "Trend / Structural Break Stationary"
            )
            .when(col("stationarity_score") == 2, "Strongly Stationary")
            .when(col("stationarity_score") == 1, "Weakly Stationary / Unstable")
            .otherwise("Non-Stationary")
        )\
        .drop("ADF_is_stationary", "KPSS_is_stationary")

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
