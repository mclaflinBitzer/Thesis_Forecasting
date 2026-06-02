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

# # Data Exploration & Quality Assessment Framework
# 
# ## 1. Structural Break Analysis
# 
# **Objective**
# - Determine whether historical data represents a single stable process or multiple regimes.
# 
# **Analysis**
# - Time series visualization
# - Rolling mean / rolling standard deviation
# - Structural break tests (CUSUM, Bai-Perron, Ruptures)
# 
# **Outputs**
# 
# ### Tables
# - Structural break summary
#     - Series
#     - Break date(s)
#     - Confidence level
#     - Business explanation
# 
# ### Visualizations
# - Time series with breakpoints
# - Rolling mean plots
# - Rolling variance plots
# 
# **Thesis Discussion**
# - What breaks were identified?
# - What business events explain them?
# - Should pre-break history be retained?
# 
# **Decision**
# - Candidate cutoff periods for historical data.
# 
# ---
# 
# ## 2. Historical Data Cutoff Selection
# 
# **Objective**
# - Determine how much history should be retained for modeling.
# 
# **Analysis**
# - Missing value assessment
# - Data coverage analysis
# - Structural break results
# - Business relevance of older history
# 
# **Outputs**
# 
# ### Tables
# - Data retention summary
#     - Series
#     - Original start date
#     - Retained start date
#     - Years retained
#     - Justification
# 
# ### Visualizations
# - Full history with retained modeling window highlighted
# 
# **Thesis Discussion**
# - Why certain historical periods were excluded.
# 
# **Decision**
# - Final modeling window per series.
# 
# ---
# 
# ## 3. Stationarity Assessment
# 
# **Objective**
# - Determine whether transformations are required to avoid spurious correlations.
# 
# **Analysis**
# - ADF test
# - KPSS test
# - Visual trend assessment
# 
# **Outputs**
# 
# ### Tables
# - Stationarity results
#     - Series
#     - ADF Statistic
#     - ADF p-value
#     - KPSS Statistic
#     - KPSS p-value
#     - Stationary (Y/N)
# 
# ### Visualizations
# - Original series
# - Differenced series
# - YoY transformed series
# 
# **Thesis Discussion**
# - Which series are stationary?
# - Which transformations are required?
# 
# **Decision**
# - Levels vs YoY vs Differencing.
# 
# ---
# 
# ## 4. Stable Period Identification
# 
# **Objective**
# - Identify periods suitable for forecasting.
# 
# **Analysis**
# - Rolling mean stability
# - Rolling variance stability
# - Seasonal consistency
# 
# **Outputs**
# 
# ### Tables
# - Stable period summary
#     - Series
#     - Stable start date
#     - Stable end date
#     - Stable duration
# 
# ### Visualizations
# - Rolling mean
# - Rolling standard deviation
# 
# **Thesis Discussion**
# - Why the selected period is considered stable.
# 
# **Decision**
# - Final training period.
# 
# ---
# 
# ## 5. Time Series Decomposition
# 
# **Objective**
# - Understand trend, seasonality, and residual behavior.
# 
# **Analysis**
# - STL decomposition
# 
# **Outputs**
# 
# ### Tables
# - Decomposition summary
#     - Trend strength
#     - Seasonal strength
#     - Residual strength
# 
# ### Visualizations
# - Observed
# - Trend
# - Seasonal
# - Residual
# 
# **Thesis Discussion**
# - Dominant patterns present in each series.
# 
# **Decision**
# - Supports transformation and feature engineering decisions.
# 
# ---
# 
# ## 6. Exploratory Analysis
# 
# **Objective**
# - Understand target behavior prior to feature selection.
# 
# **Analysis**
# - Demand distribution
# - Variability analysis
# - Missing values
# - Growth analysis
# 
# **Outputs**
# 
# ### Tables
# - Series profile summary
#     - Mean
#     - Std Dev
#     - CV
#     - Missing %
#     - Zero %
#     - CAGR
# 
# ### Visualizations
# - Time series plots
# - Histograms
# - Boxplots
# - Seasonal heatmaps
# 
# **Thesis Discussion**
# - Demand characteristics and data quality observations.
# 
# **Decision**
# - Supports feature engineering strategy.
# 
# ---


# CELL ********************

%pip install statsmodels

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
# 
# **Part 2**
# - model_data_mapping (contains the grouping columns and table name for the topline and middle models raw data)

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

# ## Overview
# 
# Dataset and the checks to be completed
# 
# 
# **Filtered**
# 
# * Quality / Gaps
# * Distribution
# * Trend / Breaks
# * Stationarity
# 
# **Topline**
# 
# * Distribution
# * Trend / Breaks
# * Stationarity
# * Seasonality / STL
# * ACF / PACF
# * Cross Series
# 
# 
# **Middle**
# 
# * Distribution
# * Trend / Breaks
# * Stationarity
# * Seasonality / STL
# * ACF / PACF
# * Cross Series

# CELL ********************

def changepoint_detection(df, series_name):
    pdf = df.toPandas().sort_values("Date").reset_index(drop=True)
    pdf["Quantity_log"] = np.log1p(pdf["Quantity"])

    
    stl = STL(pdf["Quantity_log"], period=12).fit()
    resid = stl.resid
    resid_smooth = (pd.Series(resid).rolling(3, center=True).mean().bfill().ffill().to_numpy())

    y = (resid_smooth - resid_smooth.mean())/resid_smooth.std()

    print(series_name, len(pdf), len(y))
    algo = rpt.Pelt(model="l2").fit(y)
    breaks = algo.predict(pen=100)
    changepoint_indices = breaks[:-1]
    changepoint_dates = pdf.iloc[changepoint_indices]["Date"].tolist()

    return s, changepoint_dates


## code of implementation.
topline_df = group_and_fill_data(filtered_actuals, TOPLINE_GRP_COLS, TOPLINE_TARGET_COL)
topline_df = topline_df.withColumn("series", concat(*TOPLINE_GRP_COLS))
series_list = [row["series"] for row in topline_df.select("series").distinct().collect()]

results = []
for s in series_list:
    test = topline_df.where(col("series")==s).orderBy("Date").select("Date", "Quantity")
    results.append((changepoint_detection(test,s)))

changepoints_df = spark.createDataFrame(results, ["series","changepoint_dates"])

changepoints_df = changepoints_df.withColumn("num_changepoints", size(col("changepoint_dates")))
display(changepoints_df)

changepoints_df.select(col("changepoint_dates")).collect()[0]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Visual Exploration of time series

# CELL ********************

middle_data = spark.read.table(MIDDLE_TABLE_NAME)
middle_data_filled = middle_data.fillna(0, "Quantity")

topline_data = spark.read.table(TOPLINE_TALBE_NAME)
topline_data_filled = middle_data.fillna(0,"Quantity")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(middle_data_filled.select("Product_Category").distinct().orderBy("series"))
display(middle_data_filled.where(col("series").contains("SCROLLS")))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(topline_data_filled.select("Product_Category").distinct().orderBy("series"))
display(topline_data_filled)

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


    display(joined)

    display(df_obs_stats.groupBy("count_observations").agg(count("count_observations")))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

observation_extraction(topline_data)
observation_extraction(middle_data)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # 2. Historical Cutoff Date Selection
# 
# #### Rolling stats for cutoff date selection/identification

# CELL ********************

def time_series_stats(df, group_cols, target_col):      
    w = Window.partitionBy(group_cols).orderBy("Date").rowsBetween(-12,0)

    df_stats = (
        df.withColumn("rolling_mean", avg(target_col).over(w))\
        .withColumn("prev_mean", lag("rolling_mean").over(Window.partitionBy(group_cols).orderBy("Date")))\
        .withColumn("mean_change", abs(col("rolling_mean")-col("prev_mean")))\
        .withColumn("obs_count", count("Quantity").over(w))
    )

    thresholds = (
                    df_stats.groupBy(group_cols)\
                    .agg(percentile_approx("mean_change", .5).alias("threshold"))
    )

    df_w_thresholds = df_stats.join(thresholds, on=group_cols, how="left")
    df_stable = df_w_thresholds.withColumn("is_stable", (col("mean_change") <= col("threshold")).cast("int"))
    df_stable = df_stable.withColumn("stable_run", sum("is_stable").over(Window.partitionBy(group_cols).orderBy("Date").rowsBetween(-12,0)))

    return df_stable

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

top_df = group_and_fill_data(filtered_actuals,TOPLINE_GRP_COLS, TOPLINE_TARGET_COL)
df_stats = time_series_stats(top_df, TOPLINE_GRP_COLS, TOPLINE_TARGET_COL)
df_stats = df_stats.withColumn("longest_stable_run", 
                                max(col("stable_run")).over(Window.partitionBy(TOPLINE_GRP_COLS)))
display(df_stats)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_with_cutoff = df_stats.withColumn("min_cutoff_date",
                                    min(when(col("stable_run")==col("longest_stable_run"), col("Date")))\
                                    .over(Window.partitionBy(TOPLINE_GRP_COLS)))\
                        .withColumn("max_cutoff_date",
                                    max(when(col("stable_run")==col("longest_stable_run"), col("Date")))\
                                    .over(Window.partitionBy(TOPLINE_GRP_COLS)))
display(df_with_cutoff.select("Product_Category","Date","stable_run","longest_stable_run","min_cutoff_date","max_cutoff_date"))

display(df_with_cutoff.select("Product_Category","longest_stable_run","min_cutoff_date","max_cutoff_date").distinct())


display(df_stats.groupBy(TOPLINE_GRP_COLS).agg(max(col("stable_run"))))


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

# CELL ********************

filtered_df = spark.read.table(FILTERED_TABLE_NAME)
display(filtered_df.limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Data Quality / Completeness
# missing value audit (gaps, zero inflation, structural NaNs)
# series length distribution

# CELL ********************

def run_data_quality_check(df, grouping_cols, target_col, date_col="Date"):
   
    ## Missing values
    ## how many rows have NULL values in the target column

    missing_summary = (
        df.groupBy(grouping_cols)\
            .agg(sum(col(target_col).isNull().cast("int")).alias("missing_count"))
    )

    ## Zero inflation
    ## how many rows have a 0 for the value in the target column
    zero_summary = (
        df.groupBy(grouping_cols)
          .agg(sum((col(target_col) == 0).cast("int")).alias("zero_count"))
    )
    
    ## NaN detection
    ## how many rows contain NaN in the target column
    nan_summary = (
        df.groupBy(grouping_cols)
          .agg(sum(isnan(col(target_col)).cast("int")).alias("nan_count"))
    )


    ## Gaps in time series
    ## how many missing dates exist in the time series for each group
    w = Window.partitionBy(grouping_cols).orderBy(date_col)

    df_with_lag = df.withColumn("previous_date", lag(date_col).over(w))

    gap_summary = (
        df_with_lag.withColumn("gap_days", datediff(col(date_col), col("previous_date")))\
                    .filter(col("gap_days")>1)
                    .groupBy(grouping_cols)
                    .agg(count("*").alias("gap_count"))
    )

    result = (
                missing_summary
                .join(zero_summary, grouping_cols, "left")
                .join(nan_summary, grouping_cols, "left")
                .join(gap_summary, grouping_cols, "left")
                .fillna(0)
    )

    return result

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

dq_results = {}

for model_name, cfg in model_data_mapping.items():
    print(f"Running DQ checks for: {model_name}")

    df = spark.table(cfg["table_name"])

    dq_results[model_name] = run_data_quality_check(
        df=df,
        grouping_cols=cfg["grouping_cols"],
        target_col=cfg["target_col"],
        date_col="Date"
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

dq_results['middle'].select("Product_Category","Region").distinct().count()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(dq_results['topline'].orderBy(desc("missing_count")).limit(35))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Distribution Analysis
# Histogram + KDE of target values
# log scale


# CELL ********************

topline_table = model_data_mapping['topline']['table_name']
topline_group_cols = model_data_mapping['topline']['grouping_cols']
topline_target_col = model_data_mapping['topline']['target_col']

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline = spark.read.table(topline_table)
groups = topline.select(topline_group_cols).distinct().collect()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

for row in groups:

    filters = [col(c) == row[c] for c in topline_group_cols]
    subdf = df.filter(reduce(lambda a, b: a & b, filters))

    pdf = subdf.select(topline_target_col).toPandas()
    
    # compute log transform
    pdf["log_target"] = np.log1p(pdf[topline_target_col].clip(lower=0) + 1e-6)
    
    # plot
    sns.histplot(pdf["log_target"], kde=True)
    plt.title(f"Log Distribution for {row.asDict()}")
    plt.show()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

plt.figure(figsize=(10,6))

sns.histplot(
    pdf["log_target"],
    kde=True,
    bins=50,
    color="steelblue"
)

plt.title("Histogram + KDE of Target Variable (Log Scale)")
plt.xlabel("log(Quantity + 1)")
plt.ylabel("Frequency")

plt.show()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Stationarity
# ADF
# KPSS
# Phillips Perron

# CELL ********************

topline_group_cols

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

groups

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

test = groups[0][0]

df_filter = topline.filter(col(topline_group_cols[0])==test)
display(df_filter.limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

adf_results.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting/Data_Exploration/Topline_ADF")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

results = []

for row in groups:

    product_category = row[0]

    series = (
        topline
        .filter(col(topline_group_cols[0]) == product_category)
        .select(topline_target_col)
        .toPandas()[topline_target_col]
        .dropna()
    )

    adf_stat, p_value, *_ = adfuller(series)

    results.append((
        product_category,
        adf_stat,
        p_value,
        "Stationary" if p_value < 0.05 else "Non-Stationary"
    ))

results_clean = [
    (
        str(r[0]),
        float(r[1]),
        float(r[2]),
        str(r[3])
    )
    for r in results
]

adf_schema = StructType([
    StructField("Product_Category", StringType(), False),
    StructField("ADF_Statistic", DoubleType(), False),
    StructField("P_Value", DoubleType(), False),
    StructField("Stationarity", StringType(), False)
])

adf_results = spark.createDataFrame(
    results_clean, schema=adf_schema)

return adf_results


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

for row in groups:
    df_filter = topline.filter(col(topline_group_cols[0]) == row[0])
    target = df_filter.select(topline_target_col)

    pdf = target.toPandas()
    
    series = pdf[topline_target_col].dropna()
    result = adfuller(series)

    print(f"Current Product Category is: {row[0]}")
    print(f"the result of the stationary test is: {result[0]}")
    print(f"p-value: {result[1]}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Seasonality & Decomposition
# STL decomposition

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Series Analysis
# series classification: stable / trending / seasonal / intermittent
