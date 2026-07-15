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

df = spark.read.table('Sales_Forecasting.bronze.Raw_Analyse_Sales_BPC')
display(df.limit(100))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Data Exploration & Quality Assessment Framework
# 
# ## Phase 1: Historical Data Evaluation
# 
# Purpose:
# Determine what historical data should be retained for forecasting.
# 
# ---
# 
# ## 1. Structural Break Analysis
# 
# ### Objective
# Identify major regime changes within the historical series.
# 
# ### Analysis
# - Time series visualization
# - Rolling mean
# - Rolling standard deviation
# - CUSUM test
# - Bai-Perron test
# - PELT / Ruptures
# 
# ### Outputs
# 
# #### Tables
# - Series
# - Break date(s)
# - Test statistic
# - Confidence level
# - Business explanation
# 
# #### Visualizations
# - Time series with breakpoints
# - Rolling mean
# - Rolling standard deviation
# 
# ### Thesis Discussion
# - Were structural breaks detected?
# - What business events explain them?
# - Are multiple regimes present?
# 
# ### Decision
# - Candidate break dates.
# 
# ---
# 
# ## 2. Stable Period Identification
# 
# ### Objective
# Identify periods exhibiting consistent statistical behavior.
# 
# ### Analysis
# - Rolling mean stability
# - Rolling variance stability
# - Seasonal consistency
# - Structural break alignment
# 
# ### Outputs
# 
# #### Tables
# - Series
# - Stable start date
# - Stable end date
# - Stable duration
# 
# #### Visualizations
# - Rolling mean
# - Rolling variance
# - Stability overlays
# 
# ### Thesis Discussion
# - Which periods exhibit stable behavior?
# - Which periods should be avoided?
# 
# ### Decision
# - Candidate stable periods.
# 
# ---
# 
# ## 3. Historical Data Cutoff Selection
# 
# ### Objective
# Determine final modeling history.
# 
# ### Inputs
# - Structural break results
# - Stable period analysis
# - Missing value analysis
# - Data coverage assessment
# - Business relevance assessment
# 
# ### Outputs
# 
# #### Tables
# - Series
# - Original start date
# - Retained start date
# - Years retained
# - Justification
# 
# #### Visualizations
# - Full history
# - Final retained window highlighted
# 
# ### Thesis Discussion
# - Why historical periods were removed.
# - Impact of cutoff decisions.
# 
# ### Decision
# - Final modeling window.
# 
# ---
# 
# At this point, all subsequent analysis uses only the retained modeling window.
# 
# ---
# 
# ## Phase 2: Time Series Understanding
# 
# Purpose:
# Understand the characteristics of the retained series.
# 
# ---
# 
# ## 4. Exploratory Analysis
# 
# ### Objective
# Understand statistical characteristics of the target series.
# 
# ### Analysis
# - Missing values
# - Demand distribution
# - Variability analysis
# - Growth analysis
# - Intermittency analysis
# 
# ### Outputs
# 
# #### Tables
# - Mean
# - Std Dev
# - CV
# - Missing %
# - Zero %
# - CAGR
# 
# #### Visualizations
# - Time series plots
# - Histograms
# - Boxplots
# 
# ### Thesis Discussion
# - Demand behavior
# - Data quality observations
# 
# ### Decision
# - Supports transformation and feature engineering.
# 
# ---
# 
# ## 5. Time Series Decomposition
# 
# ### Objective
# Understand trend and seasonality structure.
# 
# ### Analysis
# - STL decomposition
# 
# ### Outputs
# 
# #### Tables
# - Trend strength
# - Seasonal strength
# - Residual strength
# 
# #### Visualizations
# - Observed
# - Trend
# - Seasonal
# - Residual
# 
# ### Thesis Discussion
# - Is trend dominant?
# - Is seasonality dominant?
# - How much unexplained noise exists?
# 
# ### Decision
# - Supports transformation strategy.
# 
# ---
# 
# ## 6. Stationarity Assessment
# 
# ### Objective
# Determine whether transformations are required.
# 
# ### Analysis
# - ADF test
# - KPSS test
# - Trend assessment
# 
# ### Outputs
# 
# #### Tables
# - ADF statistic
# - ADF p-value
# - KPSS statistic
# - KPSS p-value
# - Stationary flag
# 
# #### Visualizations
# - Original series
# - YoY transformed series
# - Differenced series
# 
# ### Thesis Discussion
# - Which series require transformation?
# - Which transformation is appropriate?
# 
# ### Decision
# - Levels vs YoY vs Differencing.


# CELL ********************

%pip install statsmodels ruptures

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

%pip install arch

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

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
from statsmodels.stats.diagnostic import breaks_cusumolsresid
import statsmodels.api as sm
import ruptures as rpt
from statsmodels.tsa.stattools import kpss
from arch.unitroot import PhillipsPerron
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from matplotlib.ticker import PercentFormatter

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# this run the "Config" file where pipeline variables are set, making them available for reference and usage
# 
# the imported variables include
# 
# **Part 1**
# - _FABRIC_COLUMN_RENAME
# - _FABRIC_FILTER_COLS
# - PRODUCT_CATEGORIES
# - RAW_TABLE_NAME
# - FILTERED_TABLE_NAME
# - REGION_MAPPING_TABLE_NAME
# - CUSUM_TABLE
# - ROLLING_STATS_TABLE
# - RAW_OBS
# - STABLE_STATS_TABLE 
# - STABLE_RUNS_TABLE 
# - BASE_DQ_TABLE 
# - HISTORICAL_CUTOFF_DATES
# 
# 
# **Part 2**
# - model_data_mapping (contains the grouping columns and table name for the topline and middle models raw data)
# 
# - TOPLINE_CUTOFF_DATA_TABLE
# - MIDDLE_CUTOFF_DATA_TABLE
# - CUTOFF_DQ_TABLE
# - STL_METRICS_TABLE
# - ADF_STATS_TABLE 
# - KPSS_STATS_TABLE 
# - PP_STATS_TABLE 
# - STATIONARY_STATS_TABLE
# - HIST_CUTOFF_DATE_TABLE 

# CELL ********************

%run "Data_Exploration_Config"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# extracting topline configurations
TOPLINE_GRP_COLS = model_data_mapping["topline"]["grouping_cols"]
TOPLINE_TABLE_NAME = model_data_mapping["topline"]["table_name"]
TOPLINE_TARGET_COL = model_data_mapping["topline"]["target_col"]

# extracting middle configuration
MIDDLE_GRP_COLS = model_data_mapping["middle"]["grouping_cols"]
MIDDLE_TABLE_NAME = model_data_mapping["middle"]["table_name"]
MIDDLE_TARGET_COL = model_data_mapping["middle"]["target_col"]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # 1. Structural Break / Changepoint Detection

# MARKDOWN ********************

# ### Visual Exploration of time series

# CELL ********************

## Read Middle & Topline table data
middle_data = spark.read.table(MIDDLE_TABLE_NAME)
topline_data = spark.read.table(TOPLINE_TABLE_NAME)

## Store the distinct series of both middle & topline for later use
middle_group = middle_data.select("series").distinct()
topline_group = topline_data.select("series").distinct()

all_groups = middle_group.unionByName(topline_group)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def series_data_exploration(df, target_col):
    df = df.fillna(0, target_col)

    w_12 = (Window.partitionBy("series").orderBy("Date").rowsBetween(-11,0))
    df = df.withColumn("Rolling_mean", avg(col("Quantity")).over(w_12))
    df = df.withColumn("Rolling_std", stddev(col("Quantity")).over(w_12))
    
    ## visualizations
    groups = df.select("series").distinct().collect()
    for group in groups:
        serie = group[0]
        filtered = df.filter(col("series")==serie)
        display(filtered)

    return df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

middle_exploration = series_data_exploration(middle_data, MIDDLE_TARGET_COL)
topline_exploration = series_data_exploration(topline_data, TOPLINE_TARGET_COL)

all_exploration = topline_exploration.unionByName(middle_exploration.select("series","Date","Quantity","Rolling_mean","Rolling_std"), True)

all_exploration.write.format("delta").mode("overwrite").saveAsTable(ROLLING_STATS_TABLE)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### CUSUM test
# detects gradual drift

# CELL ********************

def cusum_test(df, target_col):

    results = []

    groups = df.select("series").distinct().collect()
    for group in groups:
        serie = group[0]
        filtered = df.filter(col("series")==serie).orderBy("Date")
        
        filtered_df = filtered.select(target_col).toPandas()
        series = filtered_df[target_col].dropna()

        if len(series) < 20:
            continue

        X = sm.add_constant(np.arange(len(series)))

        model = sm.OLS(series, X).fit()

        stat, p_value, crit = breaks_cusumolsresid(
            model.resid,
            ddof=int(model.df_model)
        )

        results.append((
            serie,
            float(stat),
            float(p_value)
        ))

    return results

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

middle_cusum = cusum_test(middle_data, MIDDLE_TARGET_COL)
topline_cusum = cusum_test(topline_data, TOPLINE_TARGET_COL)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

all_cusum = middle_cusum + topline_cusum

cusum_df_schema = StructType([
    StructField("series", StringType(), True),
    StructField("cusum_residuals_stat", DoubleType(), True),
    StructField("P_Value", DoubleType(), True)
])

cusum_df = spark.createDataFrame(all_cusum, schema=cusum_df_schema)

cusum_df = cusum_df.withColumn("Structual_Break_Flag", when(col("P_Value")<.05,"Yes").otherwise("No"))

joined_cusum = all_groups.join(cusum_df, "series", how='leftouter')

joined_cusum = joined_cusum.withColumn("Structual_Break_Flag", when(col("P_Value").isNull(),"Not enough data to process").otherwise(col("Structual_Break_Flag")))

joined_cusum.write.format("delta").mode("overwrite").saveAsTable(CUSUM_TABLE)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Pelt test
# detects breakpoints

# CELL ********************

def pelt_test(
    df,
    target_col="Quantity",
    penalty=10
):

    results = []

    groups = (
        df
        .select("series")
        .distinct()
        .collect()
    )

    for group in groups:

        serie = group[0]

        filtered = (
            df
            .filter(col("series") == serie)
            .orderBy("Date")
        )

        pdf = filtered.select(
            "Date",
            target_col
        ).toPandas()

        pdf = pdf.dropna(subset=[target_col]).reset_index(drop=True)


        signal = (
            pdf[target_col]
            .values
        )

        print(f"{serie} has {len(signal)} observations")

        if len(signal) < 24:
            continue

        algo = rpt.Pelt(
            model="rbf"
        )

        breaks = (
            algo
            .fit(signal)
            .predict(pen=penalty)
        )

        for b in breaks[:-1]:

            results.append(
                (
                    serie,
                    pdf.iloc[b]["Date"]
                )
            )

    print("-"*80)
    return results

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_pelt = pelt_test(topline_data, TOPLINE_TARGET_COL, 3)
middle_pelt = pelt_test(middle_data, MIDDLE_TARGET_COL, 3)

all_pelt_results = topline_pelt + middle_pelt

pelt_df_schema = StructType([
    StructField("series", StringType(), True),
    StructField("Structual_Change_Date", DateType(), True),
])

pelt_df = spark.createDataFrame(all_pelt_results, schema=pelt_df_schema)

all_pelt_df = all_groups.join(pelt_df, "series", how="leftouter")

all_pelt_df = all_pelt_df.withColumn("Pelt_Flag", when(col("Structual_Change_Date").isNull(),"Pelt alg identified no change point").otherwise("Pelt identified change point"))

all_pelt_df.write.format("delta").mode("overwrite").saveAsTable(PELT_TABLE)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Observation Exploration

# CELL ********************

def observation_extraction(df):
    df_obs_stats = df.groupBy("series").agg(count("Quantity").alias("count_observations"))
    global_stats = df_obs_stats.agg(
                                            min("count_observations").alias("min_observation"),
                                            max("count_observations").alias("max_observation"),
                                            avg("count_observations").alias("avg_observation")
                                        )

    joined = df_obs_stats.crossJoin(global_stats)


    joined = joined.withColumn("observation_pct", round(col("count_observations")/col("max_observation"),2))

    

    return joined

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_observations = observation_extraction(topline_data)
middle_observations = observation_extraction(middle_data)

all_observations_raw = topline_observations.unionByName(middle_observations)
all_observations_raw.write.format("delta").mode("overwrite").saveAsTable(RAW_OBS)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # 2. Stable Period Identification

# CELL ********************

def time_series_stats(df, group_cols, target_col):      
    w = Window.partitionBy(group_cols).orderBy("Date").rowsBetween(-11,0)

    df_stats = (
        df.withColumn("rolling_mean_12", avg(target_col).over(w))\
        .withColumn("prev_mean", lag("rolling_mean_12").over(Window.partitionBy(group_cols).orderBy("Date")))\
        .withColumn("mean_change", abs(col("rolling_mean_12")-col("prev_mean")))\
        .withColumn("obs_count", count("Quantity").over(w))
    )

    thresholds = (
                    df_stats.groupBy(group_cols)\
                    .agg(percentile_approx("mean_change", .5).alias("threshold"))
    )

    df_w_thresholds = df_stats.join(thresholds, on=group_cols, how="left")
    df_stable = df_w_thresholds.withColumn("is_stable", (col("mean_change") <= col("threshold")).cast("int"))
    df_stable = df_stable.withColumn("stable_run", sum("is_stable").over(Window.partitionBy(group_cols).orderBy("Date").rowsBetween(-12,0)))


    df_stable = df_stable.withColumn("longest_stable_run", 
                                    max(col("stable_run")).over(Window.partitionBy(*group_cols)))


    df_stable = df_stable.withColumn("min_cutoff_date",
                                        min(
                                            when(
                                                col("stable_run")==col("longest_stable_run"), 
                                                add_months(col("Date"), -col("longest_stable_run"))
                                            )
                                        ).over(Window.partitionBy(*group_cols)))\
                            .withColumn("max_cutoff_date",
                                        max(
                                            when(
                                                col("stable_run")==col("longest_stable_run"), 
                                                add_months(col("Date"), -col("longest_stable_run"))
                                            )
                                        ).over(Window.partitionBy(*group_cols)))

    df_stable = df_stable.select("series", 'Date', 'Quantity', 'rolling_mean_12', 'prev_mean', 'mean_change', 
                                'obs_count', 'threshold', 'is_stable', 'stable_run', 'longest_stable_run', 
                                'min_cutoff_date', 'max_cutoff_date')

    stable_runs = df_stable
    stable_runs = stable_runs.filter(col("stable_run")==col("longest_stable_run"))

    stable_runs = stable_runs.withColumn("stable_run_start_date", add_months(col("Date"), -col("longest_stable_run")))

    stable_runs = stable_runs.select("series","Date","longest_stable_run","stable_run_start_date")

    stable_runs = stable_runs.orderBy(col("series"),col("stable_run_start_date").asc())

    return df_stable, stable_runs

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_run_stats, topline_stable_runs = time_series_stats(topline_data, TOPLINE_GRP_COLS, TOPLINE_TARGET_COL)

middle_run_stats, middle_stable_runs = time_series_stats(middle_data, MIDDLE_GRP_COLS, MIDDLE_TARGET_COL)

all_run_stats = topline_run_stats.unionByName(middle_run_stats)
all_stable_runs = topline_stable_runs.unionByName(middle_stable_runs)

all_run_stats.write.format("delta").mode("overwrite").saveAsTable(STABLE_STATS_TABLE)
all_stable_runs.write.format("delta").mode("overwrite").saveAsTable(STABLE_RUNS_TABLE)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # 3. Historical Cutoff Date Selection
# * missing value assessment
# * data coverage analysis
# * latest date allowed set based on data needed for models
# * cutoff date setting per series

# MARKDOWN ********************

# ### Data Quality / Completeness
# missing value audit (gaps, zero inflation, structural NaNs), series lengths distribution

# CELL ********************

def run_data_quality_check(df, target_col, date_col="Date"):
   
    ## Missing values
    ## how many rows have NULL values in the target column

    missing_summary = (
        df.groupBy("series")\
            .agg(sum(col(target_col).isNull().cast("int")).alias("null_count"))
    )

    ## Zero inflation
    ## how many rows have a 0 for the value in the target column
    zero_summary = (
        df.groupBy("series")
          .agg(sum((col(target_col) == 0).cast("int")).alias("zero_count"))
    )
    
    ## NaN detection
    ## how many rows contain NaN in the target column
    nan_summary = (
        df.groupBy("series")
          .agg(sum(isnan(col(target_col)).cast("int")).alias("nan_count"))
    )


    ## Gaps in time series
    ## how many missing dates exist in the time series for each group
    w = Window.partitionBy("series").orderBy(date_col)

    df_with_lag = df.withColumn("previous_date", lag(date_col).over(w))

    gap_summary = (
        df_with_lag.withColumn("gap_months", months_between(col(date_col), col("previous_date")))\
                    .filter(col("gap_months")>1)
                    .groupBy("series")
                    .agg(count("*").alias("gap_count"))
    )

    result = (
                missing_summary
                .join(zero_summary, "series", "left")
                .join(nan_summary, "series", "left")
                .join(gap_summary, "series", "left")
                .fillna(0)
    )

    return result

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_data_quality = run_data_quality_check(topline_data,TOPLINE_TARGET_COL)
middle_data_quality = run_data_quality_check(middle_data, MIDDLE_TARGET_COL)

all_data_quality = topline_data_quality.unionByName(middle_data_quality)

all_data_quality.write.format("delta").mode("overwrite").saveAsTable(BASE_DQ_TABLE)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Cutoff Date Assignment
# 
# set within the Config files

# CELL ********************

# HISTORICAL_CUTOFF_DATES = {
#     "ALU": "01-03-2019",
#     "AVP_CDU": "01-05-2016",
#     "HEXPV": "01-01-2018",
#     "MAERSK_COMPRESSOR": "01-02-2020",
#     "MAERSK_ELECTRONICS": "01-10-2019",
#     "PISTON": "01-10-2016",
#     "SCREWS": "01-04-2020",
#     "SCROLLS": "01-06-2015"
# }

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

historical_cutoff_dates = [
    ["ALU", "01-03-2019"],
    ["AVP_CDU", "01-05-2016"],
    ["HEXPV", "01-01-2018"],
    ["MAERSK_COMPRESSOR", "01-02-2020"],
    ["MAERSK_ELECTRONICS", "01-10-2019"],
    ["PISTON", "01-10-2016"],
    ["SCREWS", "01-04-2020"],
    ["SCROLLS", "01-06-2015"]
]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd

df_cutoffs = pd.DataFrame(
    historical_cutoff_dates,
    columns=["series", "historical_cutoff_date"]
)

df_cutoffs["historical_cutoff_date"] = pd.to_datetime(
                                df_cutoffs["historical_cutoff_date"],
                                format="%d-%m-%Y"
                            )


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

HIST_CUTOFF_DATE_TABLE = "Sales_Forecasting.silver.historical_cutoff_dates"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

spark_df_cutoff_dates = spark.createDataFrame(df_cutoffs)

spark_df_cutoff_dates.write.format("delta").mode("overwrite").saveAsTable(HIST_CUTOFF_DATE_TABLE)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Use data w/ cutoff date applied

# CELL ********************

## creating dataframe w/ the cutoff dates
data = []
for key, item in HISTORICAL_CUTOFF_DATES.items():
    data.append({
                "series":key,
                "cutoff_date":item
                })
df = pd.DataFrame(data)
cutoff_dates = spark.createDataFrame(df)
cutoff_dates = cutoff_dates.withColumn("cutoff_date", to_date(col("cutoff_date"), "dd-MM-yyyy"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

filtered_topline_data = (
    topline_data.alias("d")
    .join(
        cutoff_dates.alias("c"),
        col("d.series").startswith(col("c.series")),
        "inner"
    )
    .filter(col("d.Date")>=col("c.cutoff_date"))
    .select("d.*")
)
display(filtered_topline_data.limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

filtered_middle_data = (
    middle_data.alias("d")
    .join(
        cutoff_dates.alias("c"),
        col("d.series").startswith(col("c.series")),
        "inner"
    )
    .filter(col("d.Date")>=col("c.cutoff_date"))
    .select("d.*")
)
display(filtered_middle_data.limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

filtered_topline_data.write.format("delta").mode("overwrite").saveAsTable(TOPLINE_CUTOFF_DATA_TABLE)
filtered_middle_data.write.format("delta").mode("overwrite").saveAsTable(MIDDLE_CUTOFF_DATA_TABLE)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # 4. Exploratory Analysis - Cutoff filtered data
# understand target behavior prior to feature selection


# CELL ********************

topline_cutoff_data = spark.read.table(TOPLINE_CUTOFF_DATA_TABLE)
middle_cutoff_data = spark.read.table(MIDDLE_CUTOFF_DATA_TABLE)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Data Quality Check

# CELL ********************

cutoff_topline_dq = run_data_quality_check(topline_cutoff_data,TOPLINE_TARGET_COL)
cutoff_middle_dq = run_data_quality_check(middle_cutoff_data, MIDDLE_TARGET_COL)

all_cutoff_dq = cutoff_topline_dq.unionByName(cutoff_middle_dq)

all_cutoff_dq.write.format("delta").mode("overwrite").saveAsTable(CUTOFF_DQ_TABLE)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # 5. Time Series Decomposition
# * STL decomposition
# * table of trend, seasonal, residual data
# * visualizations of original, trend, seasonal, residual data

# CELL ********************

def STL_decomposition(df):
    ## create list of all series that the STL analysis needs to be conducted on
    series = df.select("series").distinct().collect()

    ## impute null values with 0 for STL alg
    df = df.fillna({"Quantity":0})

    ## create list to capture all STL metrics 
    metrics = []
    for row in series:

        ## create pandas dataframe for each unique time series
        serie = row["series"]
        series_df = df.filter(col("series")==serie).orderBy("Date")
        series_pdf = series_df.select("Date", "Quantity").toPandas()

        ## preping the time series for STL processing
        series_pdf["Date"] = pd.to_datetime(series_pdf["Date"])

        series_pdf = series_pdf.set_index("Date")

        ts = series_pdf["Quantity"].asfreq("MS")

        ## applying STL
        stl = STL(
            ts,
            period=12,
            robust=True
        )

        result = stl.fit()

        ## visualizing results
        fig = result.plot()

        fig.set_size_inches(12,8)

        fig.suptitle(f"{serie} - STL Decomposition", fontsize=16, fontweight="bold", y=1.02)


        ## Writing the plot to the lakehouse 
        file_name = f"{serie}_STL.png"
        local_path = f"/tmp/{file_name}"
        lakehouse_path = f"Files/Visualizations/STL/{file_name}"

        fig.savefig(local_path, bbox_inches="tight")
        plt.close(fig)

        mssparkutils.fs.cp(
            f"file:{local_path}",
            lakehouse_path
        )


        trend = result.trend
        seasonal = result.seasonal
        resid = result.resid

        eps = 1e-10

        seasonal_denom = (seasonal + resid).var()
        trend_denom = (trend + resid).var()

        seasonal_strength = float(builtins.max(0, 1 - resid.var() / (seasonal_denom + eps)))
        trend_strength = float(builtins.max(0, 1 - resid.var() / (trend_denom + eps)))

        noise_ratio = float(resid.var() / (seasonal.var() + trend.var() + resid.var() + eps))

        metrics.append({
                        "series":serie,
                        "seasonal_strength": seasonal_strength,
                        "trend_strength": trend_strength,
                        "noise_ratio": noise_ratio
                        })

    return spark.createDataFrame(metrics)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_STL_metrics = STL_decomposition(topline_cutoff_data)
middle_STL_metrics = STL_decomposition(middle_cutoff_data)


all_STL_metrics = topline_STL_metrics.unionByName(middle_STL_metrics).select("series","seasonal_strength","trend_strength", "noise_ratio")


# seasonal and trend strength metric
# | Strength | Interpretation       |
# | -------- | -------------------- |
# | >0.8     | Strong    |
# | 0.4–0.8  | Moderate  |
# | <0.4     | Weak      |

## residual variance metric
# | Value   | Interpretation    |
# | >0.5    | High Noise        |
# | >=0.2   | Moderate Noise    |
# | <0.2    | Low Noise         |
all_STL_metrics = all_STL_metrics.withColumn("seasonal_strength_flag", 
                                            when(col("seasonal_strength")>0.8,"Strong")
                                            .when(col("seasonal_strength")>=0.4, "Moderate")
                                            .otherwise("Weak")
                                            )\
                                .withColumn("trend_strength_flag",
                                            when(col("trend_strength")>=0.8, "Strong")
                                            .when(col("trend_strength")>=0.4,"Moderate")
                                            .otherwise("Weak")
                                            )\
                                .withColumn("noise_level_flag",
                                            when(col("noise_ratio")>0.5, "High Noise")
                                            .when(col("noise_ratio")>=0.2, "Moderate Noise")
                                            .otherwise("Low Noise"))

                                            
all_STL_metrics.write.format("delta").mode("overwrite").saveAsTable(STL_METRICS_TABLE)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # 6. Stationarity
# ADF, KPSS, Phillips Perron

# MARKDOWN ********************

# #### ADF test

# CELL ********************

def adf_test(df):

    results = []

    series = df.select("series").distinct().collect()

    for row in series:

        serie = row["series"]

        serie_df = (
            df
            .filter(col("series") == serie)
            .orderBy("Date")
            .toPandas()
        )

        # ensure clean numeric series
        ts = (
            serie_df["Quantity"]
            .astype(float)
            .dropna()
            .values
        )

        # -----------------------------------------
        # NEW: minimum sample size guard
        # -----------------------------------------
        if len(ts) < 12:
            results.append({
                "series": serie,
                "adf_stat": None,
                "p_value": None,
                "stationary_flag": "Insufficient Data (<12 obs)"
            })
            continue

        # -----------------------------------------
        # Safe ADF execution
        # -----------------------------------------
        try:
            adf_stat, p_value, *_ = adfuller(ts)

            results.append({
                "series": serie,
                "adf_stat": float(adf_stat),
                "p_value": float(p_value),
                "stationary_flag": (
                    "Strongly Stationary" if p_value < 0.01
                    else "Stationary" if p_value < 0.05
                    else "Weakly Non-Stationary" if p_value < 0.10
                    else "Non-Stationary"
                )
            })

        except Exception as e:
            results.append({
                "series": serie,
                "adf_stat": None,
                "p_value": None,
                "stationary_flag": f"ADF Failed: {str(e)}"
            })

    # -----------------------------------------
    # Conversion to Spark
    # -----------------------------------------
    adf_schema = StructType([
        StructField("Series", StringType(), False),
        StructField("ADF_Statistic", DoubleType(), True),
        StructField("P_Value", DoubleType(), True),
        StructField("Stationarity_Flag", StringType(), False)
    ])

    adf_results = spark.createDataFrame(
        [
            (
                r["series"],
                r["adf_stat"],
                r["p_value"],
                r["stationary_flag"]
            )
            for r in results
        ],
        schema=adf_schema
    )

    return adf_results

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

middle_adf_results = adf_test(middle_cutoff_data)
topline_adf_results = adf_test(topline_cutoff_data)
   

all_adf_results = topline_adf_results.unionByName(middle_adf_results)

all_adf_results.write.format("delta").mode("overwrite").saveAsTable(ADF_STATS_TABLE)

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

    series = df.select("series").distinct().collect()

    for row in series:

        serie = row["series"]

        serie_df = (
            df
            .filter(col("series") == serie)
            .orderBy("Date")
            .toPandas()
        )

        ts = (
            serie_df["Quantity"]
            .astype(float)
            .dropna()
            .values
        )

        n_obs = len(ts)

        # --------------------------------------------------
        # Minimum sample size guardrail
        # --------------------------------------------------
        if n_obs < 12:

            results.append({
                "series": serie,
                "n_obs": n_obs,
                "kpss_stat": None,
                "p_value": None,
                "stationarity_flag": "Insufficient Data (<12 obs)"
            })

            continue

        # --------------------------------------------------
        # Run KPSS
        # --------------------------------------------------
        try:

            kpss_stat, p_value, _, _ = kpss(
                ts,
                regression="c",      # level stationarity
                nlags="auto"
            )

            results.append({
                "series": serie,
                "n_obs": n_obs,
                "kpss_stat": float(kpss_stat),
                "p_value": float(p_value),
                # KPSS NULL = Stationary

                "stationarity_flag":
                    "Strongly Stationary" if p_value >= 0.10
                    else "Stationary" if p_value >= 0.05
                    else "Weakly Non-Stationary" if p_value >= 0.01
                    else "Non-Stationary"
            })

        except Exception as e:

            results.append({
                "series": serie,
                "n_obs": n_obs,
                "kpss_stat": None,
                "p_value": None,
                "stationarity_flag": f"KPSS Failed: {str(e)}"
            })

    # --------------------------------------------------
    # Create Spark DataFrame
    # --------------------------------------------------

    kpss_schema = StructType([
        StructField("Series", StringType(), False),
        StructField("N_Obs", IntegerType(), False),
        StructField("KPSS_Statistic", DoubleType(), True),
        StructField("P_Value", DoubleType(), True),
        StructField("Stationarity_Flag", StringType(), False)
    ])

    kpss_results = spark.createDataFrame(
        [
            (
                r["series"],
                r["n_obs"],
                r["kpss_stat"],
                r["p_value"],
                r["stationarity_flag"]
            )
            for r in results
        ],
        schema=kpss_schema
    )

    return kpss_results

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_kpss = kpss_test(topline_cutoff_data)
middle_kpss = kpss_test(middle_cutoff_data)

all_kpss_stats =  topline_kpss.unionByName(middle_kpss)

all_kpss_stats.write.format("delta").mode("overwrite").saveAsTable(KPSS_STATS_TABLE)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Phillips Perron test

# CELL ********************

def pp_test(df):

    results = []

    series = df.select("series").distinct().collect()

    for row in series:

        serie = row["series"]

        serie_df = (
            df
            .filter(col("series") == serie)
            .orderBy("Date")
            .toPandas()
        )

        ts = (
            serie_df["Quantity"]
            .astype(float)
            .dropna()
            .values
        )

        n_obs = len(ts)

        if n_obs < 12:

            results.append({
                "series": serie,
                "n_obs": n_obs,
                "pp_stat": None,
                "p_value": None,
                "stationarity_flag": "Insufficient Data (<12 obs)"
            })

            continue

        try:

            pp = PhillipsPerron(ts)

            pp_stat = float(pp.stat)
            p_value = float(pp.pvalue)

            results.append({
                "series": serie,
                "n_obs": n_obs,
                "pp_stat": pp_stat,
                "p_value": p_value,
                "stationarity_flag":
                    "Strongly Stationary" if p_value < 0.01
                    else "Stationary" if p_value < 0.05
                    else "Weakly Non-Stationary" if p_value < 0.10
                    else "Non-Stationary"
            })

        except Exception as e:

            results.append({
                "series": serie,
                "n_obs": n_obs,
                "pp_stat": None,
                "p_value": None,
                "stationarity_flag": f"PP Failed: {str(e)}"
            })

    schema = StructType([
        StructField("Series", StringType(), False),
        StructField("N_Obs", IntegerType(), False),
        StructField("PP_Statistic", DoubleType(), True),
        StructField("P_Value", DoubleType(), True),
        StructField("Stationarity_Flag", StringType(), False)
    ])

    return spark.createDataFrame(
        [
            (
                r["series"],
                r["n_obs"],
                r["pp_stat"],
                r["p_value"],
                r["stationarity_flag"]
            )
            for r in results
        ],
        schema=schema
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_pp_results = pp_test(topline_cutoff_data)
middle_pp_results = pp_test(middle_cutoff_data)

all_pp_results = topline_pp_results.unionByName(middle_pp_results)

all_pp_results.write.format("delta").mode("overwrite").saveAsTable(PP_STATS_TABLE)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Merging all Stationary test for final assignment

# CELL ********************

adf_stats = spark.read.table(ADF_STATS_TABLE)
kpss_stats = spark.read.table(KPSS_STATS_TABLE)
pp_stats = spark.read.table(PP_STATS_TABLE)

adf_stats_clean = adf_stats.withColumnsRenamed({"P_Value":"ADF_P_Value", "Stationarity_Flag":"ADF_Stationarity_Flag"})
kpss_stats_clean = kpss_stats.withColumnsRenamed({"P_Value":"KPSS_P_Value","Stationarity_Flag":"KPSS_Stationarity_Flag"}).drop("N_Obs")
pp_stats_clean = pp_stats.withColumnsRenamed({"P_Value":"PP_P_Value", "Stationarity_Flag":"PP_Stationarity_Flag"})

all_stats = adf_stats_clean\
                .join(kpss_stats_clean, "Series")\
                .join(pp_stats_clean, "Series")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df = all_stats  

# -----------------------------
# Step 1: Binary stationarity signals
# -----------------------------
df = (
    df
    .withColumn("ADF_is_stationary", when(col("ADF_P_Value") < 0.05, 1).otherwise(0))
    .withColumn("PP_is_stationary", when(col("PP_P_Value") < 0.05, 1).otherwise(0))
    .withColumn("KPSS_is_stationary", when(col("KPSS_P_Value") > 0.05, 1).otherwise(0))
)

# -----------------------------
# Step 2: Composite score
# -----------------------------
df = df.withColumn(
    "stationarity_score",
    col("ADF_is_stationary") +
    col("PP_is_stationary") +
    col("KPSS_is_stationary")
)

# -----------------------------
# Step 3: Final stationarity flag (base logic + conflict handling)
# -----------------------------
df = df.withColumn(
            "Final_Stationarity_Flag",
            when(
                (col("ADF_is_stationary") == 1) &
                (col("PP_is_stationary") == 1) &
                (col("KPSS_is_stationary") == 0),
                "Trend / Structural Break Stationary"
            )
            .when(col("stationarity_score") == 3, "Strongly Stationary")
            .when(col("stationarity_score") == 2, "Stationary (Mixed Evidence)")
            .when(col("stationarity_score") == 1, "Weakly Stationary / Unstable")
            .otherwise("Non-Stationary")
        )\
        .drop("ADF_is_stationary","PP_is_stationary", "KPSS_is_stationary","stationarity_score")


df.write.format("delta").mode("overwrite").saveAsTable(STATIONARY_STATS_TABLE)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Stationarity Visual Check
# - original series (Date vs Quantity)
#     
#     look for mean or variance drift, as well as sructural breaks 
# - Date vs YoY Growth 
#     
#     look for stable oscillation around zero (stable), or persistent trends (non stationary)
# - First difference (Yt - Yt-1)
# 
#     look for random fluctuation around zero (stable), or persistent trends (non stationary)
# - Rolling Stats visualization (original, rolling mean, rolling std)
#     
#     look for flat rolling mean/variance (stable), non flat (non stationary)

# CELL ********************

def stationary_visualization(df):
    series = df.select("series").distinct().collect()

    for row in series:

        ## create pandas dataframe for each unique time series
        serie = row["series"]
        series_df = all_stationarity.filter(col("series")==serie).orderBy("Date")
        series_pdf = series_df.select("Date", "Quantity","YoY_Growth","first_diff","rolling_mean","rolling_std").toPandas()

        ## Date on x axis
        series_pdf["Date"] = pd.to_datetime(series_pdf["Date"])
        series_pdf = series_pdf.set_index("Date")


        # -----------------------------
        # Plot 1: Quantity + Stationarity Metrics
        # -----------------------------

        fig1, ax1 = plt.subplots(figsize=(12, 8))

        series_pdf[
            [
                "Quantity",
                "first_diff",
                "rolling_mean",
                "rolling_std"
            ]
        ].plot(ax=ax1)

        ax1.set_title(f"{serie} - Quantity, First Difference, Rolling Mean & Rolling Std")
        ax1.set_xlabel("Date")

        # More y-axis ticks
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=15))

        # Minor ticks between major ticks
        ax1.yaxis.set_minor_locator(AutoMinorLocator())

        # Grid lines
        ax1.grid(True, which="major", linestyle="--", alpha=0.7)
        ax1.grid(True, which="minor", linestyle=":", alpha=0.4)
        ax1.axhline(y=0, linestyle="--", alpha=0.5)

        plt.tight_layout()



        ## Writing the plot to the lakehouse 
        file_name = f"{serie}_Stationarity_Metrics.png"
        local_path = f"/tmp/{file_name}"
        lakehouse_path = f"Files/Visualizations/Stationarity_Metrics/{file_name}"

        fig1.savefig(local_path, bbox_inches="tight")
        plt.close(fig1)

        mssparkutils.fs.cp(
            f"file:{local_path}",
            lakehouse_path
        )

        # -----------------------------
        # Plot 2: YoY Growth
        # -----------------------------


        fig2, ax2 = plt.subplots(figsize=(12, 8))

        series_pdf[["YoY_Growth"]].plot(ax=ax2)

        ax2.set_title(f"{serie} - YoY Growth")
        ax2.set_xlabel("Date")

        # More y-axis ticks
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=12))

        # Show as percentages
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))

        # Grid lines
        ax2.grid(True, which="major", linestyle="--", alpha=0.7)
        ax2.grid(True, which="minor", linestyle=":", alpha=0.4)
        ax2.axhline(y=0, linestyle="--", alpha=0.5)

        plt.tight_layout()



        ## Writing the plot to the lakehouse 
        file_name = f"{serie}_YoY_Growth.png"
        local_path = f"/tmp/{file_name}"
        lakehouse_path = f"Files/Visualizations/Stationarity_Metrics/{file_name}"

        fig2.savefig(local_path, bbox_inches="tight")
        plt.close(fig2)

        mssparkutils.fs.cp(
            f"file:{local_path}",
            lakehouse_path
        )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

all_cutoff_data = topline_cutoff_data.select("series","Date","Quantity").unionByName(middle_cutoff_data.select("series","Date","Quantity"))
w = Window.partitionBy("series").orderBy("Date")

w_12 = (Window.partitionBy("series").orderBy("Date").rowsBetween(-11,0))


all_stationarity = all_cutoff_data\
                            .withColumn("lag_12", lag(col("Quantity"),12).over(w))\
                            .withColumn(
                                            "YoY_Growth", 
                                            ((col("Quantity")-col("lag_12"))/col("lag_12"))
                                        )\
                            .withColumn("first_diff", col("Quantity")-lag(col("Quantity"),1).over(w))\
                            .withColumn("rolling_mean", avg(col("Quantity")).over(w_12))\
                            .withColumn("rolling_std", stddev(col("Quantity")).over(w_12))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

stationary_visualization(all_stationarity)

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
