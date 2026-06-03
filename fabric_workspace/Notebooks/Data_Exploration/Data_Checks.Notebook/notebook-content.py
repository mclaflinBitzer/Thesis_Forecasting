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

# # 2. Stable Period Identification

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
top_df = group_and_fill_data(filtered_actuals,TOPLINE_GRP_COLS, TOPLINE_TARGET_COL)
df_stats = time_series_stats(top_df, TOPLINE_GRP_COLS, TOPLINE_TARGET_COL)
df_stats = df_stats.withColumn("longest_stable_run", 
                                max(col("stable_run")).over(Window.partitionBy(TOPLINE_GRP_COLS)))
display(df_stats)
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

display(dq_results['topline'].orderBy(desc("missing_count")).limit(35))
display(dq_results['middle'].orderBy(desc("missing_count")))

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

# # Use data w/ cutoff date applied

# MARKDOWN ********************

# # 4. Exploratory Analysis
# understand target behavior prior to feature selection


# MARKDOWN ********************

# #### Distribution Analysis 
# Histogram + KDE of target values log scaled

# CELL ********************

def histogram_viz(groups, group_cols df):

    for row in groups:

        filters = [col(c) == row[c] for c in group_cols]
        subdf = df.filter(reduce(lambda a, b: a & b, filters))

        pdf = subdf.select(topline_target_col).toPandas()
        
        # compute log transform
        pdf["log_target"] = np.log1p(pdf[topline_target_col].clip(lower=0) + 1e-6)
        
        # plot
        sns.histplot(pdf["log_target"], kde=True)
        plt.title(f"Log Distribution for {row.asDict()}")
        plt.show()
histogram_viz(topline_groups, TOPLINE_GRP_COLS, topline)
histogram_viz(middle_groups, MIDDLE_GRP_COLS, middle)
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

# # 5. Time Series Decomposition
# * STL decomposition
# * table of trend, seasonal, residual data
# * visualizations of original, trend, seasonal, residual data

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

# # 6. Stationarity
# ADF, KPSS, Phillips Perron

# CELL ********************

def adf_test(groups, group_cols, df):
    results = []

    for row in groups:

        record = row[0]

        series = (
            df
            .filter(col(group_cols[0]) == record)
            .select(group_cols)
            .toPandas()[group_cols]
            .dropna()
        )

        adf_stat, p_value, *_ = adfuller(series)

        results.append((
            record,
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

middle_adf_results = adf_test(middle_groups, MIDDLE_GRP_COLS, middle)
topline_adf_results = adf_test(topline_groups, MIDDLE_GRP_COLS, topline)

display(middle_adf_results)
display(topline_adf_results)    

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

middle_adf_results.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting/Data_Exploration/Middle_ADF")
topline_adf_results.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting/Data_Exploration/Topline_ADF")

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
