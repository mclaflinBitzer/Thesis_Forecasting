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
    sheet_name="Project Value Spread_GD_Sectors",
    header=None
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# -----------------------------------------------------------------------------
# REMOVE TOP 4 ROWS
# -----------------------------------------------------------------------------

df_temp = df_raw.iloc[5:].reset_index(drop=True)


# -----------------------------------------------------------------------------
# FIRST HEADER PROMOTION
# -----------------------------------------------------------------------------

df_temp.columns = df_temp.iloc[0]

df_df_tempstep1 = df_temp.iloc[1:].reset_index(drop=True)


# -----------------------------------------------------------------------------
# OPTIONAL: CLEAN COLUMN NAMES
# -----------------------------------------------------------------------------

df_temp.columns = [
    str(col).strip()
    for col in df_temp.columns
]


df_temp = df_temp.iloc[1:, 2:]



# -----------------------------------------------------------------------------
# IDENTIFY YEAR COLUMNS
# -----------------------------------------------------------------------------

year_cols = [
    col for col in df_temp.columns
    if str(col).isdigit()
]

# -----------------------------------------------------------------------------
# CONVERT YEAR COLUMNS TO NUMERIC
# -----------------------------------------------------------------------------

for col in year_cols:
    df_temp[col] = pd.to_numeric(
        df_temp[col],
        errors="coerce"
    )



# -----------------------------------------------------------------------------
# UNPIVOT / MELT
# -----------------------------------------------------------------------------

df_unpivot = df_temp.melt(
    id_vars=["Country", "Sector"],
    value_vars=year_cols,
    var_name="Year",
    value_name="Value"
)


# -----------------------------------------------------------------------------
# RENAME COLUMNS
# -----------------------------------------------------------------------------

df_unpivot = df_unpivot.rename(
    columns={
        "Sector": "Indicator",
        "Year": "Date"
    }
)


# -----------------------------------------------------------------------------
# CONVERT DATE COLUMN
# -----------------------------------------------------------------------------

df_unpivot["Date"] = pd.to_datetime(
    df_unpivot["Date"].astype(str) + "-01-01"
)

# -----------------------------------------------------------------------------
# CONVERT VALUE COLUMN
# -----------------------------------------------------------------------------

df_unpivot["Value"] = pd.to_numeric(
    df_unpivot["Value"],
    errors="coerce"
)


df_unpivot.head(10)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## Creating "world" grouping
df_world = df_unpivot.groupby(["Indicator","Date"], as_index=False)["Value"].sum()

df_world["Country"] = "World"
df_world = df_world[["Country","Indicator","Date","Value"]]


# -----------------------------------------------------------------------------
# APPEND EXISTING WORLD DATAFRAME
# Assumes Project_GD_World already exists
# -----------------------------------------------------------------------------

final_df = pd.concat(
    [df_unpivot, df_world],
    ignore_index=True
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## Creating spark schema
spark_schema = StructType([
    StructField("Country", StringType(), False),
    StructField("Indicator", StringType(), False),
    StructField("Date", DateType(), False),
    StructField("Value", DoubleType(), True)
])

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

display(spark_df.orderBy("Country", "Indicator", "Date"))
display(filled_df.orderBy("Country", "Indicator", "Date"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

filled_df.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.bronze.DRV_Sector")

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
