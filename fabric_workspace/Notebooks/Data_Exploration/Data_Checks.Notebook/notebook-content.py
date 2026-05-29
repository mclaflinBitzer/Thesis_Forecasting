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

from pyspark.sql.functions import *
from pyspark.sql.window import Window

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

%run "Data_Exploration_Config"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

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

from pyspark.sql.functions import col
from functools import reduce

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

import seaborn as sns
import matplotlib.pyplot as plt

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

%pip install statsmodels

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from statsmodels.tsa.stattools import adfuller


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

# MARKDOWN ********************

# ### Series Analysis
# series classification: stable / trending / seasonal / intermittent

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
