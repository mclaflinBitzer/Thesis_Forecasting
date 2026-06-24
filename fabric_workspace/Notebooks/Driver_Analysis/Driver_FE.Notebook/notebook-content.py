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

# ### Stage 1: Feature Engineering
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
# ### Stage 2: Driver Classification
# 
# **Purpose:** Select correct transformations
# - apply ADF and KPSS
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
# 
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
from statsmodels.tsa.stattools import ccf 
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

compiled_drivers = spark.read.table("Sales_Forecasting.silver.compiled_drivers").select("Country","Indicator","Region","Date","Value")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 1. Feature Engineering
# For each Driver create the following using original data
# - Rolling (3, 6, 12) std, mean
# - Growth (YoY, MoM, diff)
# 
#         input: original / raw driver data
#         output: engineered features

# CELL ********************

## Feature Transformations to do
    ## levels
    ## Rolling Mean / STD (3,6,12)
    ## Growth: YoY, MoM, diff

## levels
drivers_FE = compiled_drivers.withColumn("level", col("Value"))

## Window creation
w = Window.partitionBy("Region", "Country","Indicator").orderBy("Date")
w_3 = Window.partitionBy("Region", "Country","Indicator").orderBy("Date").rowsBetween(-2,0)
w_6 = Window.partitionBy("Region", "Country", "Indicator").orderBy("Date").rowsBetween(-5,0)
w_12 = Window.partitionBy("Region", "Country", "Indicator").orderBy("Date").rowsBetween(-11,0)

## Rolling Means
drivers_FE = drivers_FE.withColumn("rolling_mean_3", avg("Value").over(w_3))
drivers_FE = drivers_FE.withColumn("rolling_mean_6", avg("Value").over(w_6))
drivers_FE = drivers_FE.withColumn("rolling_mean_12", avg("Value").over(w_12))

## Rolling STD
drivers_FE = drivers_FE.withColumn("rolling_std_3", stddev("Value").over(w_3))
drivers_FE = drivers_FE.withColumn("rolling_std_6", stddev("Value").over(w_6))
drivers_FE = drivers_FE.withColumn("rolling_std_12", stddev("Value").over(w_12))


## Growth: YoY, MoM, Diff
drivers_FE = drivers_FE.withColumn("diff", (col("Value") - lag("Value", 1).over(w)))
drivers_FE = drivers_FE.withColumn("YoY_pct", 
                        when(lag("Value",12).over(w) == 0, None)
                        .otherwise((col("Value")/lag("Value",12).over(w)) -1))
drivers_FE = drivers_FE.withColumn("MoM_pct",
                        when(lag("Value",1).over(w)==0, None)
                        .otherwise((col("Value") / lag("Value", 1).over(w))-1))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Unpivoting the Features in order to apply ADF & KPSS testing

# CELL ********************

drivers_FE_unpivot = drivers_FE.select("Country","Indicator","Region", "Date",
    expr("""
        stack(10,
        'level', level,
        'rolling_mean_3', rolling_mean_3,
        'rolling_mean_6', rolling_mean_6,
        'rolling_mean_12', rolling_mean_12,
        'rolling_std_3', rolling_std_3,
        'rolling_std_6', rolling_std_6,
        'rolling_std_12', rolling_std_12,
        'diff', diff,
        'YoY_pct', YoY_pct,
        'MoM_pct', MoM_pct
        ) as (Feature, Value)
    """)
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

drivers_FE_unpivot.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.silver.feature_set")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Stage 2: Driver Classification
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
    region = pdf["Region"].iloc[0]
    feature = pdf["Feature"].iloc[0]

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
            "Region": region,
            "Feature": feature,
            "adf_stat": None,
            "adf_p_value": None,
            "adf_stationary_flag": "Insufficient Data"            
        }])
    
    if np.nanstd(ts) == 0:
        return pd.DataFrame([{
            "Country": country,
            "Indicator": indicator,
            "Region": region,
            "Feature": feature,
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
            "Region": region,
            "Feature": feature,
            "adf_stat": adf_stat,
            "adf_p_value": p_value,
            "adf_stationary_flag": (
                "Stationary" if p_value < 0.05
                else "Non Stationary"
            )
        }])

    except Exception as e:
        return pd.DataFrame([{
            "Country": country,
            "Indicator": indicator,
            "Region": region,
            "Feature": feature,
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
    StructField("Country", StringType(), False),
    StructField("Indicator", StringType(), False),
    StructField("Region", StringType(), False),
    StructField("Feature", StringType(), False),
    StructField("adf_stat", DoubleType(), True),
    StructField("adf_p_value", DoubleType(), True),
    StructField("adf_stationary_flag", StringType(), True)
])

adf_results = (drivers_FE_unpivot
                    .groupBy("Country","Indicator","Region","Feature")
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
    region = pdf["Region"].iloc[0]
    feature = pdf["Feature"].iloc[0]

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
            "Region": region,
            "Feature": feature,
            "kpss_stat": None,
            "kpss_p_value": None,
            "kpss_stationary_flag": "Insufficient Data (<12 obs)"
        }])
    
    # SAFETY CHECK 2: constant series
    if np.nanstd(ts) == 0:
        return pd.DataFrame([{
            "Country": country,
            "Indicator": indicator,
            "Region": region,
            "Feature": feature,
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
                "Region": region,
                "Feature": feature,
                "kpss_stat": kpss_stat,
                "kpss_p_value": p_value,
                "kpss_stationary_flag": 
                    "Stationary" if p_value >= 0.05
                    else "Non-Stationary"
        }])


    except Exception as e:
        return pd.DataFrame([{
                "Country": country,
                "Indicator": indicator,
                "Region": region,
                "Feature": feature,
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
                    StructField("Region", StringType(), False),
                    StructField("Feature", StringType(), False),
                    StructField("kpss_stat", DoubleType(), True),
                    StructField("kpss_p_value", DoubleType(), True),
                    StructField("kpss_stationary_flag", StringType(), False)
])

kpss_results = (drivers_FE_unpivot.groupBy("Country", "Indicator","Region", "Feature")
                    .applyInPandas(kpss_group, schema=kpss_schema))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Combine adf & kpss outputs
# - combine outputs
# - apply valid or invalid flag
# - write to driver stats table 
# - remove records where either adf or kpss has skipped, flagged for insufficient data, or the test has failed
# - create binary flag for stationary or requires differencing


# CELL ********************

df_stationary_stats = adf_results.join(kpss_results, ["Country", "Indicator","Region","Feature"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## add valid/invalid flag
    ## if valid then 1 otherwise 0
df_stationary_stats = df_stationary_stats.withColumn("valid_flag", 
    when(
        (
            (col("adf_stationary_flag").isin("Constant Series (Skipped)", "ADF Failed: Invalid input, x is constant", "Insufficient Data")) |
            (col("kpss_stationary_flag").isin("Constant Series (Skipped)", "Insufficient Data (<12 obs)","KPSS Failed: cannot convert float infinity to integer"))
        ), lit(0)
    ).otherwise(lit(1))
    
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

driver_final_stats = df_stationary_stats.withColumn(
    "ADF_is_stationary",
    when(col("valid_flag")==0, None)
    .when(col("adf_p_value") < 0.05, 1)
    .otherwise(0)
)

driver_final_stats = driver_final_stats.withColumn(
    "KPSS_is_stationary",
    when(col("valid_flag")==0, None)
    .when(col("kpss_p_value") > 0.05, 1)
    .otherwise(0)
)


## if both tests identified the driver as stationary then use directly for CCF otherwise apply differencing transformation
driver_final_stats = driver_final_stats.withColumn(
        "stationary_class",
        when((col("ADF_is_stationary")==1) & (col("KPSS_is_stationary")==1), "Stationary_use_levels")
        .when((col("ADF_is_stationary")==1) & (col("KPSS_is_stationary")==0), "Detrend")
        .otherwise("Log_diff")
    )\
    .drop("ADF_status","KPSS_status","ADF_is_stationary","KPSS_is_stationary")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

driver_final_stats.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration.driver_stats")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Stage 2: Target Transformation
# **Purpose:** Align target + drivers
# ##### Transformations - Apply STL in order to only do the correlation analysis upon the residuals
# ---

# MARKDOWN ********************

# ## Apply STL to all drivers and series before CCF analysis

# CELL ********************

feature_set = spark.read.table("Sales_Forecasting.silver.feature_set")
topline_cutoff_data = spark.read.table("Sales_Forecasting.silver.topline_cutoff_data").withColumnRenamed("Quantity","Value")
middle_cutoff_data = spark.read.table("Sales_Forecasting.silver.middle_cutoff_data").withColumnRenamed("Quantity","Value")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

driver_classification = spark.read.table("Sales_Forecasting.Driver_Exploration.driver_stats")\
        .drop('adf_stat',
            'adf_p_value',
            'adf_stationary_flag',
            'kpss_stat',
            'kpss_p_value',
            'kpss_stationary_flag',)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

feature_set_valid = broadcast(
    driver_classification.filter(col("valid_flag")==1)
).join(
    feature_set, ["Country","Indicator","Region","Feature"], "inner"
).drop("valid_flag","stationary_class")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def stl_decompose(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf = pdf.sort_values("Date")
    pdf["Value"] = pdf["Value"].replace(np.nan, 0)

    stl = STL(pdf["Value"], period=12, robust=True)
    result = stl.fit()

    #pdf["trend"] = result.trend
    #pdf["seasonal"] = result.seasonal
    pdf["residual"] = result.resid

    return pdf

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_schema = StructType([
    StructField("Product_Category", StringType(), False),
    StructField("series", StringType(), False),
    StructField("Date", DateType(), False),
    StructField("Value", DoubleType(), True),
    StructField("residual", DoubleType(), True)
])

topline_residuals = topline_cutoff_data.groupBy("Product_Category","series")\
        .applyInPandas(
            stl_decompose,
            schema=topline_schema
        )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

middle_schema = StructType([
    StructField("Product_Category", StringType(), False),
    StructField("Region", StringType(), False),
    StructField("series", StringType(), False),
    StructField("Date", DateType(), False),
    StructField("Value", DoubleType(), True),
    StructField("residual", DoubleType(), True)
])

middle_residuals = middle_cutoff_data.groupBy("Product_Category","Region","series")\
    .applyInPandas(stl_decompose, schema=middle_schema)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

feature_set_schema = StructType([
    StructField("Country", StringType(), False),
    StructField("Indicator", StringType(), False),
    StructField("Region", StringType(), False),
    StructField("Feature", StringType(), False),
    StructField("Date", DateType(), False),
    StructField("Value", DoubleType(), True),
    StructField("residual", DoubleType(), True)
])

feature_set_residuals = feature_set_valid.groupBy("Country","Indicator", "Region", "Feature")\
    .applyInPandas(stl_decompose, schema=feature_set_schema)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Stage 3: Lag/Lead Identification of Features
# #### Application of CCF

# MARKDOWN ********************

# #### Joining target / drivers prior to CCF calculation

# CELL ********************

feature_set_residuals = feature_set_residuals\
    .drop("Value")\
    .withColumnsRenamed({"Date":"feature_date", "residual":"feature_residual","Region":"feature_region"})

topline_residuals = topline_residuals\
    .drop("Value")\
    .withColumnsRenamed({"Date":"target_date","residual":"target_residual","Region":"topline_region"})
    
middle_residuals = middle_residuals\
    .drop("Value")\
    .withColumnsRenamed({"Date":"target_date","residual":"target_residual","Region":"middle_region"})


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## Create valid mapping for target-driver combinations

## create distinct records of each df set
topline_distinct = topline_residuals.select("series").distinct()

middle_distinct = middle_residuals.select("series","middle_region").distinct()

feature_world_distinct = feature_set_residuals.filter(col("feature_region")=="World").select("feature_region","Country","Indicator","Feature").distinct()
feature_m_distinct = feature_set_residuals.filter(~(col("feature_region")=="World"))\
    .select("feature_region","Country","Indicator","Feature").distinct()


## create pair dataframes
topline_pairs = broadcast(topline_distinct).crossJoin(feature_world_distinct)

display(topline_pairs.limit(3))

middle_w_pairs = broadcast(middle_distinct).crossJoin(feature_world_distinct)
middle_pairs = broadcast(middle_distinct).join(
    feature_m_distinct,
    middle_distinct["middle_region"]==feature_m_distinct["feature_region"],
    "inner"
)

middle_pairs = middle_pairs.unionByName(middle_w_pairs)

display(middle_pairs.limit(3))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_pairs.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration.topline_pairs")
middle_pairs.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration.middle_pairs")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## Creating Expanded versions of the middle & topline residuals based on the pair mappings
    ## both approaches for managing schema/column explosion during join work
    
expanded_topline = topline_residuals.join(
    broadcast(topline_pairs),
    #topline_residuals["series"]==topline_pairs["series"],
    ["series"],
    "left"
)#.drop(topline_pairs["series"])

display(expanded_topline.limit(5))

expanded_middle = middle_residuals.join(
    broadcast(middle_pairs),
    (middle_residuals["series"]==middle_pairs["series"]) & 
    (middle_residuals["middle_region"] == middle_pairs["middle_region"]),
    "left"
).drop(middle_pairs["series"],middle_pairs["middle_region"])

display(expanded_middle.limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

expanded_topline.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration.expanded_topline")
expanded_middle.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration.expanded_middle")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## Creating Expanded version of the middle & topline features based on the pair mappings
expanded_t_features = broadcast(topline_pairs).join(
    feature_set_residuals,
    ["feature_region","Indicator","Feature"],
    "inner"
).drop(feature_set_residuals["Country"])

display(expanded_t_features.limit(3))

expanded_m_features = broadcast(middle_pairs).join(
    feature_set_residuals,
    ["feature_region","Indicator","Feature","Country"],
    "inner"
)

display(expanded_m_features.limit(3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

expanded_t_features.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration.expanded_t_features")
expanded_m_features.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration.expanded_m_features")

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

feature_set_residuals = spark.read.table("Sales_Forecasting.Driver_Exploration.expanded_t_features")
expanded_t_features = feature_set_residuals
expanded_m_features = spark.read.table("Sales_Forecasting.Driver_Exploration.expanded_m_features")


expanded_topline = spark.read.table("Sales_Forecasting.Driver_Exploration.expanded_topline")
expanded_middle = spark.read.table("Sales_Forecasting.Driver_Exploration.expanded_middle")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(spark.read.table("Sales_Forecasting.Driver_Exploration.expanded_t_features").limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

expanded_topline = spark.createDataFrame(expanded_topline.rdd, expanded_topline.schema)
expanded_middle = spark.createDataFrame(expanded_middle.rdd, expanded_middle.schema)

expanded_t_features = spark.createDataFrame(expanded_t_features.rdd, expanded_t_features.schema)
expanded_m_features = spark.createDataFrame(expanded_m_features.rdd, expanded_m_features.schema)


t = expanded_topline.alias("t")
m = expanded_middle.alias("m")
tf = expanded_t_features.alias("tf")
mf = expanded_m_features.alias("mf")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

final_topline = t.join(
    tf,
    (t["series"] == tf["series"]) &
    (t["feature_region"] == tf["feature_region"]) &
    (t["Country"] == tf["Country"]) &
    (t["Indicator"] == tf["Indicator"]) &
    (t["Feature"] == tf["Feature"]) &
    (t["target_date"] == tf["feature_date"]),
    "left"    
).select(t["series"],t["Product_Category"],t["target_date"],t["target_residual"],t["feature_region"],t["Country"],
        t["Indicator"],t["Feature"],tf["feature_residual"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

final_topline.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration.topline_w_features")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

middle_final = m.join(
    mf,
    (m["series"]==mf["series"]) &
    (m["middle_region"] ==mf["middle_region"]) &
    (m["feature_region"]==mf["feature_region"]) &
    (m["Country"]==mf["Country"]) &
    (m["Indicator"]==mf["Indicator"]) &
    (m["Feature"]==mf["Feature"]) &
    (m["target_date"]==mf["feature_date"]),
    "left"
).select(m["series"],m["Product_Category"],m["middle_region"],m["target_date"],m["target_residual"],
        mf["feature_region"],mf["Indicator"],mf["Feature"],mf["Country"],mf["feature_residual"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

middle_final.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration.middle_w_features")

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

# MARKDOWN ********************

# #### CCF Compute

# CELL ********************

def compute_topline_ccf(pdf):



    pdf = pdf.sort_values("date")

    driver = pdf["driver"]
    target = pdf["target"]

    results = []


    for lag in range(-24, 25):

        if lag < 0:
            d = driver.iloc[:lag]
            t = target.iloc[-lag:]

        elif lag > 0:
            d = driver.iloc[lag:]
            t = target.iloc[:-lag]

        else:
            d = driver
            t = target

        corr = d.corr(t) if len(d) > 1 and len(t) > 1 else None


        results.append({
            "series": pdf["series"].iloc[0],
            "feature_region": pdf["feature_region"].iloc[0],
            "Country": pdf["Country"].iloc[0],
            "Indicator": pdf["Indicator"].iloc[0],
            "Feature": pdf["Feature"].iloc[0],
            "Lag": lag,
            "Correlation": (
                            float(0.0)
                            if corr is None or pd.isna(corr) or np.isinf(corr)
                            else float(corr)
                        )
        })

    return pd.DataFrame(results)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def compute_middle_ccf(pdf):



    pdf = pdf.sort_values("date")

    driver = pdf["driver"]
    target = pdf["target"]

    results = []


    for lag in range(-24, 25):

        if lag < 0:
            d = driver.iloc[:lag]
            t = target.iloc[-lag:]

        elif lag > 0:
            d = driver.iloc[lag:]
            t = target.iloc[:-lag]

        else:
            d = driver
            t = target

        corr = d.corr(t) if len(d) > 1 and len(t) > 1 else None


        results.append({
            "series": pdf["series"].iloc[0],
            "Product_Category": pdf["Product_Category"].iloc[0],
            "target_region": pdf["target_region"].iloc[0],
            "feature_region": pdf["feature_region"].iloc[0],
            "Country": pdf["Country"].iloc[0],
            "Indicator": pdf["Indicator"].iloc[0],
            "Feature": pdf["Feature"].iloc[0],
            "Lag": lag,
            "Correlation": (
                            float(0.0)
                            if corr is None or pd.isna(corr) or np.isinf(corr)
                            else float(corr)
                        )
        })

    return pd.DataFrame(results)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_w_features = spark.read.table("Sales_Forecasting.Driver_Exploration.topline_w_features")\
    .withColumnsRenamed({"target_date":"date", "target_residual":"target", "feature_residual":"driver"})\
    .select("series","Product_Category","feature_region","Country","Indicator","Feature","date","target","driver")

middle_w_features = spark.read.table("Sales_Forecasting.Driver_Exploration.middle_w_features")\
    .withColumnsRenamed({"target_date":"date", "target_residual":"target", "feature_residual":"driver", "middle_region":"target_region"})\
    .select("series","Product_Category","target_region","feature_region","Country","Indicator","Feature","date","target","driver")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_ccf_schema = (StructType([
    StructField("series", StringType(), False),
    StructField("feature_region", StringType(), False),
    StructField("Country", StringType(), False),
    StructField("Indicator", StringType(), False),
    StructField("Feature", StringType(), False),
    StructField("Lag", IntegerType(), False),
    StructField("Correlation", DoubleType(), True)
]))

topline_ccf = topline_w_features.groupBy("series","feature_region","Country","Indicator","Feature")\
    .applyInPandas(compute_topline_ccf, schema=topline_ccf_schema)

topline_ccf.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration.topline_ccf_base")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

middle_ccf_schema = (StructType([
    StructField("series", StringType(), False),
    StructField("Product_Category", StringType(), True),
    StructField("target_region", StringType(), True),
    StructField("feature_region", StringType(), True),
    StructField("Country", StringType(), True),
    StructField("Indicator", StringType(), True),
    StructField("Feature", StringType(), True),
    StructField("Lag", IntegerType(), True),
    StructField("Correlation", DoubleType(), True)
]))

middle_ccf = middle_w_features.groupBy("series","Product_Category","target_region","feature_region","Country","Indicator","Feature")\
    .applyInPandas(compute_middle_ccf, schema=middle_ccf_schema)


middle_ccf.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration.middle_ccf_base")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 4. Feature Selection
# For each time series w/ all potential driver/feature combinations
# - Correlation Filtering (Pearson/Spearman) strength threshold/ranking
# - Multicollinearity / VIF (remove redundant predictors)
# - Model importance (SHAP / Feature importance), which predictors actually help with forecasting
# 
#         input: raw driver data at lag/lead identified, engineered features at lag/lead identified 
#         output: take the top x features

# CELL ********************

topline_ccf = spark.read.table("Sales_Forecasting.Driver_Exploration.topline_ccf_base")
middle_ccf = spark.read.table("Sales_Forecasting.Driver_Exploration.middle_ccf_base")

display(topline_ccf.limit(10))
display(middle_ccf.limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def ccf_filtering(df):
    window = Window.partitionBy("series","feature_region","Country","Indicator","Feature")
    df = df.withColumn("abs_corr", abs(col("Correlation")))
    df = df.withColumn("max_corr", max(col("abs_corr")).over(window))

    ## filter out records with a correlations < .3 or Indicator is NaN
    df_filtered = df.filter((col("max_corr")>.3) & (col("Indicator") != "NaN"))

    return df_filtered

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_ccf_filtered = ccf_filtering(topline_ccf)
display(topline_ccf_filtered.limit(10))

middle_ccf_filtered = ccf_filtering(middle_ccf)
display(middle_ccf_filtered.limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_ccf_filtered.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration.topline_ccf_filtered") 
middle_ccf_filtered.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration.middle_ccf_filtered")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Correlation Clustering

# CELL ********************

def corr_clustering(df):

    serie = df['series'][0]
    print(serie)
    print(f"intial df shape {df.shape}")

    X = df.drop(columns=["target_date","series"])
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median(numeric_only=True))

    corr = X.corr().abs()
    corr = corr.fillna(0)
    corr = np.clip(corr, 0, 1)

    ## convert correlation matrix to distance matrix
    distance = 1 - corr
    distance = np.clip(distance, 0, 1)



    ## hierarchical clustering 
    condensed_dist = squareform(distance.values, checks=False)

    linkage=sch.linkage(condensed_dist, method="average")

    threshold = 0.2
    cluster_labels = sch.fcluster(linkage,t=threshold, criterion="distance")

    cluster_map = pd.DataFrame({
        "serie": serie,
        "feature_id": X.columns,
        "cluster": cluster_labels
    })

    representatives = []
    for cluster_id in cluster_map["cluster"].unique():
        features = cluster_map[cluster_map["cluster"] == cluster_id]["feature_id"].tolist()
        
        if len(features) == 1:
            representatives.append(features[0])
            continue

        sub_corr = corr.loc[features, features]

        # score = mean correlation to others (higher = more central)
        scores = sub_corr.mean(axis=1)

        rep = scores.idxmax()
        representatives.append(rep)

    X_reduced = X[representatives]

    final_df = pd.concat([df[["target_date"]], X_reduced], axis=1)
    print(f"final df shape {final_df.shape}")
    print(f"max clusters {cluster_map['cluster'].max()}")

    return cluster_map
    


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def top_20_feature_extraction(df_f_resid, df_ccf_filtered, df_):
    df_f_resid.select("series","feature_region","Country","Indicator","Feature", "target_date","feature_residual")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

middle_w_features = spark.read.table("Sales_Forecasting.Driver_Exploration.middle_w_features")
display(middle_w_features.limit(10))
topline_w_features = spark.read.table("Sales_Forecasting.Driver_Exploration.topline_w_features")
display(topline_w_features.limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_w_features = spark.read.table("Sales_Forecasting.Driver_Exploration.topline_w_features")\
    .select("series","feature_region","Country","Indicator","Feature","target_date","feature_residual")
display(topline_w_features.limit(10))

topline_ccf_filtered = spark.read.table("Sales_Forecasting.Driver_Exploration.topline_ccf_filtered")
topline_ccf_filtered = topline_ccf_filtered.withColumn("feature_serie",
    concat_ws("__","series","feature_region","Country", "Indicator", "Feature"))

series_features = topline_ccf_filtered.select("feature_serie","series","feature_region","Country","Indicator","Feature").distinct()

feature_series = series_features.select("feature_serie").distinct()
feature_series = feature_series.withColumn("feature_id",row_number().over(Window.orderBy("feature_serie")))


topline_ccf_joined = feature_series.join(topline_ccf_filtered, ["feature_serie"], "left")

display(topline_ccf_joined.limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_concat = topline_w_features.withColumn("feature_serie",
    concat_ws("__","series","feature_region","Country", "Indicator", "Feature"))

topline_joined = feature_series.join(topline_concat, ["feature_serie"], "left")


display(topline_joined.limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_wide = topline_joined.groupBy("series","target_date")\
    .pivot("feature_id").agg(first("feature_residual"))

display(topline_wide.orderBy("series",asc("target_date")).limit(50))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd
import numpy as np

import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

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

test_schema = StructType([
    StructField("serie", StringType(), False),
    StructField("feature_id", StringType(), False),
    StructField("cluster", IntegerType(), False)
])

topline_clustering = topline_wide.groupBy("series").applyInPandas(corr_clustering, schema=test_schema)
topline_clustering = topline_clustering.withColumn("feature_id", col("feature_id").cast("int"))
display(topline_clustering.orderBy('serie','feature_id').limit(20))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print(topline_ccf.columns)
print(topline_clustering.columns)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_ccf_joined = spark.createDataFrame(topline_ccf_joined.rdd, topline_ccf_joined.schema)
topline_clustering = spark.createDataFrame(topline_clustering.rdd, topline_clustering.schema)
tj = topline_ccf_joined.alias("tj")
tc = topline_clustering.alias("tc")


topline_w_cluster = tj.join(tc, 
    (tj["feature_id"]==tc["feature_id"]) & (tj['series']==tc['serie']),
    "inner").drop(tc["feature_id"])
display(topline_w_cluster.limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

cluster_corr_w = Window.partitionBy("series", "cluster").orderBy(desc("max_corr"))
topline_distinct = topline_w_cluster.select("series","feature_id","cluster","max_corr").distinct()

test = topline_distinct.withColumn("cluster_rank", row_number().over(cluster_corr_w))
display(test.filter(col("series")=="ALU"))

display(test.groupBy("series").agg(
    countDistinct(col("cluster")).alias("num_clusters"),
    countDistinct(col("feature_id")).alias("num_features")
    ))
display(test.count())
display(test.filter(col("cluster_rank")<3).count())
filtered_cluster = test.filter(col("cluster_rank")<3)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print(filtered_cluster.columns)
print(topline_ccf_joined.columns)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## join remaining features w/ original data

#display(topline_ccf_joined.limit(3))
fc = filtered_cluster.alias("fc")
tc = topline_ccf_joined.alias("tc")

filtered_post_cluster = fc.join(tc, ["feature_id"], "inner").drop(fc['series'], fc['max_corr'])

#display(filtered_post_cluster)

ranked_f_w = Window.partitionBy("series").orderBy(desc("max_corr"))
filtered_post_cluster = filtered_post_cluster.withColumn("feature_rank", dense_rank().over(ranked_f_w))


topline_20_features = filtered_post_cluster.filter(col("feature_rank")<=10)\
    .select("series","feature_region","Country","Indicator","Feature","Lag","Correlation","abs_corr","max_corr","feature_rank")
topline_20_features = topline_20_features.withColumn("rec_lag", 
    when(col("abs_corr")==col("max_corr"),lit(1)).otherwise(lit(0)))

topline_20_features.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration.topline_20_features")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(
    filtered_post_cluster.orderBy("series","feature_rank")\
    .select("series","feature_id","cluster","cluster_rank","feature_serie","Lag","max_corr","feature_rank")
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Testing w/ Middle schema

# CELL ********************

middle_w_features = spark.read.table("Sales_Forecasting.Driver_Exploration.middle_w_features")\
    .select("series","feature_region","Country","Indicator","Feature","target_date","feature_residual")

middle_ccf_filtered

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

df = topline_wide.toPandas()
X = df.drop(columns=["target_date","series"])
X = X.select_dtypes(include=[np.number])
X = X.fillna(X.median(numeric_only=True))

corr = X.corr().abs()
corr = corr.fillna(0)
corr = np.clip(corr, 0, 1)

## convert correlation matrix to distance matrix
distance = 1 - corr
distance = np.clip(distance, 0, 1)



## hierarchical clustering 
condensed_dist = squareform(distance.values, checks=False)

linkage=sch.linkage(condensed_dist, method="average")

threshold = 0.2
cluster_labels = sch.fcluster(linkage,t=threshold, criterion="distance")

cluster_map = pd.DataFrame({
    "feature": X.columns,
    "cluster": cluster_labels
})

representatives = []
for cluster_id in cluster_map["cluster"].unique():
    features = cluster_map[cluster_map["cluster"] == cluster_id]["feature"].tolist()
    
    if len(features) == 1:
        representatives.append(features[0])
        continue

    sub_corr = corr.loc[features, features]

    # score = mean correlation to others (higher = more central)
    scores = sub_corr.mean(axis=1)

    rep = scores.idxmax()
    representatives.append(rep)

X_reduced = X[representatives]

final_df = pd.concat([df[["target_date"]], X_reduced], axis=1)

display(final_df)
display(cluster_map.sort_values("cluster"))
print(representatives)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print(final_df.shape)
print(df.shape)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Reduce feature set based on colinearity 

# CELL ********************

display(feature_series.count())
filtered_features = feature_series.filter((col("feature_id").isin(representatives)))
display(filtered_features.count())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Mapping / Selecting on representative Features

# CELL ********************

df = df[final_df.columns]
rep_features = spark.createDataFrame(df)
display(rep_features.limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df.columns

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(topline_joined.filter(col("feature_id").isin(rep_features.columns)))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# 
# 
# 
# 
# ### 5. Complete Economic Validation
# - validate choices with Dominik for feature selection
# 
# ### 5. Save the target time series / feature selection as output
# 
# 
# ### 6. Input target time series & top x selected Features into Models & Forecast


# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Stage 3: Feature Engineering
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
