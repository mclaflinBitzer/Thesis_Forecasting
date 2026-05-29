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

import pandas as pd
import numpy as np
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.functions import col
from pyspark.sql.window import Window

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# -----------------------------------------------------------------------------
# FILE PATH
# -----------------------------------------------------------------------------

file_path = (
    "/lakehouse/default/Files/Driver_Data/"
    "Global_Data_Project_Data.xlsx"
)

# -----------------------------------------------------------------------------
# READ EXCEL SHEET
# -----------------------------------------------------------------------------

df_raw = pd.read_excel(
    file_path,
    sheet_name="Project Value Spread_HVAC",
    header=None
)

# -----------------------------------------------------------------------------
# REMOVE TOP 4 ROWS
# -----------------------------------------------------------------------------

df_raw = df_raw.iloc[5:].reset_index(drop=True)

# -----------------------------------------------------------------------------
# SECOND HEADER PROMOTION
# -----------------------------------------------------------------------------

df_raw.columns = df_raw.iloc[0]

df_raw = df_raw.iloc[1:, 2:].reset_index(drop=True)


# -----------------------------------------------------------------------------
# CLEAN COLUMN NAMES
# -----------------------------------------------------------------------------

df_raw.columns = [
    str(col).strip()
    for col in df_raw.columns
]
df_raw.columns = [
    str(col).replace(".0", "").strip()
    for col in df_raw.columns
]

# -----------------------------------------------------------------------------
# IDENTIFY YEAR COLUMNS
# -----------------------------------------------------------------------------

year_cols = [
    col for col in df_raw.columns
    if col.isdigit()
]

# -----------------------------------------------------------------------------
# CONVERT YEAR COLUMNS TO NUMERIC
# -----------------------------------------------------------------------------

for col in year_cols:
    df_raw[col] = pd.to_numeric(
        df_raw[col],
        errors="coerce"
    )


# -----------------------------------------------------------------------------
# UNPIVOT / MELT
# -----------------------------------------------------------------------------

df_unpivot = df_raw.melt(
    id_vars=["Country", "Sector", "Sub Sector"],
    value_vars=year_cols,
    var_name="Date",
    value_name="Value"
)

# -----------------------------------------------------------------------------
# CONVERT DATE COLUMN
# -----------------------------------------------------------------------------

df_unpivot["Date"] = pd.to_datetime(
    df_unpivot["Date"].astype(str).str.replace(".0","") + "-01-01"
)

# -----------------------------------------------------------------------------
# CONVERT VALUE COLUMN
# -----------------------------------------------------------------------------

df_unpivot["Value"] = pd.to_numeric(
    df_unpivot["Value"],
    errors="coerce"
)

# -----------------------------------------------------------------------------
# GROUP BY COUNTRY + DATE
# Aggregates all sectors/subsectors into HVAC indicator
# -----------------------------------------------------------------------------

df_grouped = (
    df_unpivot
    .groupby(
        ["Country", "Date"],
        as_index=False
    )["Value"]
    .sum()
)

# -----------------------------------------------------------------------------
# ADD INDICATOR COLUMN
# -----------------------------------------------------------------------------

df_grouped["Indicator"] = "HVAC"

# -----------------------------------------------------------------------------
# REORDER COLUMNS
# -----------------------------------------------------------------------------

df_grouped = df_grouped[
    ["Country", "Indicator", "Date", "Value"]
]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## create world grouping
df_world = df_grouped.groupby(["Date"],as_index=False)["Value"].sum()
df_world["Country"] = "World"
df_world["Indicator"] = "HVAC"

df_world = df_world[["Country", "Indicator", "Date", "Value"]]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# -----------------------------------------------------------------------------
# APPEND WORLD DATAFRAME
# Assumes HVAC_GD_World already exists
# -----------------------------------------------------------------------------

final_df = pd.concat(
    [df_grouped, df_world],
    ignore_index=True
)

final_df.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

spark_schema = StructType([
                    StructField("Country", StringType(), False),
                    StructField("Indicator", StringType(), False),
                    StructField("Date", DateType(), False),
                    StructField("Value", DoubleType(), True)
])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

spark_df = spark.createDataFrame(final_df, schema=spark_schema)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

driver_time_bounds = (
                        spark_df
                                .filter(col("Value").isNotNull() & ~isnan(col("Value")))
                                .groupBy("Country","Indicator")
                                .agg(
                                    min("Date").alias("min_date"),
                                    add_months(max("Date"),12).alias("max_date")
                                )
)

## date grid of the min date -> max date with monthly steps
date_grid = driver_time_bounds.select("Country", "Indicator",\
                                        explode(sequence(
                                            trunc("min_date","MM"),
                                            trunc("max_date", "MM"),
                                            expr("interval 1 month")
                                        )).alias("Date")
                                    )

## join the date grid w/ the actual data
df_full = date_grid.join(spark_df, on=["Country","Indicator","Date"],how="left")


filled_df = df_full.withColumn(
    "Value",
    last("Value", ignorenulls=True).over(
        Window.partitionBy("Country", "Indicator")
              .orderBy("Date")
              .rowsBetween(Window.unboundedPreceding, 0)
    )
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

filled_df.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.bronze.DRV_HVAC")

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
