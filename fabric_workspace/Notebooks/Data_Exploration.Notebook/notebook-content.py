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
import pandas as pd
from pyspark.sql.types import *
from pyspark.sql.utils import AnalysisException
from pyspark.sql.window import Window
import numpy as np

## commented out as this method wasn't working
# %pip install ruptures
# import ruptures as rpt

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

%run Config

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

# ## Changepoint detection method
# won't be implemented due to lack of success in making meaningful changepoints

# CELL ********************

from statsmodels.tsa.seasonal import STL

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

# ## Rolling stats for cutoff date selection

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

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(df_with_cutoff.select("Product_Category","longest_stable_run","min_cutoff_date","max_cutoff_date").distinct())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(df_stats.groupBy(TOPLINE_GRP_COLS).agg(max(col("stable_run"))))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Visual Data Exploration of time series

# CELL ********************

middle_data = spark.read.table(MIDDLE_TABLE_NAME)
middle_data_filled = middle_data.fillna(0, "Quantity")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(middle_data_filled.select("Product_Category").distinct().orderBy("series"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(middle_data_filled.where(col("series").contains("SCROLLS")))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Observation exploration 

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

topline_data = spark.read.table(TOPLINE_TABLE_NAME)
observation_extraction(topline_data)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

raw_data = spark.read.table("Sales_Forecasting.bronze.filtered_data")
display(raw_data.limit(1))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print(f"number of unique product categories {raw_data.select('Product_Category').distinct().count()}")
print(f"number of unique regions: {raw_data.select('Region').distinct().count()}")
print(f"number of unique product category x region combinations: {raw_data.select('Region','Product_Category').distinct().count()}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(raw_data.select("Product_Category","Region").distinct())

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
