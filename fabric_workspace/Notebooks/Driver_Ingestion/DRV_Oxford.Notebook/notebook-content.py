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
import re

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
    "Oxecon download - 9 April 2026 10_45_44.xlsx"
)

# -----------------------------------------------------------------------------
# READ EXCEL SHEET
# -----------------------------------------------------------------------------

df_read = pd.read_excel(
    file_path,
    sheet_name="Default",
    header=0
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_raw = df_read.copy()

# -----------------------------------------------------------------------------
# CLEAN COLUMN NAMES
# -----------------------------------------------------------------------------

df_raw.columns = [
    str(col).strip()
    for col in df_raw.columns
]

# -----------------------------------------------------------------------------
# IDENTIFY QUARTER COLUMNS (e.g. 2010Q1)
# -----------------------------------------------------------------------------

quarter_cols = [
    col for col in df_raw.columns
    if re.match(r"^\d{4}Q[1-4]$", str(col))
]


# -----------------------------------------------------------------------------
# REPLACE "NA" AND 0 VALUES
# -----------------------------------------------------------------------------

df_raw[quarter_cols] = df_raw[quarter_cols].replace(
    ["NA", 0],
    np.nan
)

# -----------------------------------------------------------------------------
# RENAME CORE COLUMNS
# -----------------------------------------------------------------------------

df_raw = df_raw.rename(columns={
    "Location": "Country",
    "Indicator": "original_indicator"
})

# -----------------------------------------------------------------------------
# CREATE COMPOSITE INDICATOR
# Indicator = Sector + "_" + original_indicator
# -----------------------------------------------------------------------------

df_raw["Indicator"] = df_raw["Sector"].astype(str) + "_" + df_raw["original_indicator"].astype(str)


# -----------------------------------------------------------------------------
# UNPIVOT (MELT)
# -----------------------------------------------------------------------------

df_unpivot = df_raw.melt(
    id_vars=["Country", "Indicator", "Indicator code"],
    value_vars=quarter_cols,
    var_name="Date",
    value_name="Value"
)

# -----------------------------------------------------------------------------
# CREATE DATE FROM QUARTER
# Example: 2010Q2 -> 2010-04-01
# -----------------------------------------------------------------------------

def quarter_to_date(q):
    year = int(q[:4])
    quarter = int(q[-1])
    month = (quarter - 1) * 3 + 1
    return pd.Timestamp(year=year, month=month, day=1)

df_unpivot["Date"] = df_unpivot["Date"].apply(quarter_to_date)

# -----------------------------------------------------------------------------
# CLEAN VALUE COLUMN
# -----------------------------------------------------------------------------

df_unpivot["Value"] = pd.to_numeric(df_unpivot["Value"], errors="coerce")


# -----------------------------------------------------------------------------
# FINAL ORDER
# -----------------------------------------------------------------------------

df_unpivot = df_unpivot[
    ["Country", "Indicator", "Indicator code", "Date", "Value"]
]

df_unpivot = df_unpivot.rename(columns={"Indicator code":"Indicator_Code"})

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_unpivot.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_world = df_unpivot.groupby(["Indicator", "Indicator_Code", "Date"], as_index=False)["Value"].sum()
df_world["Country"] = "World"

df_world = df_world[["Country", "Indicator", "Indicator_Code", "Date", "Value"]]

df_world.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# -----------------------------------------------------------------------------
# APPEND OXFORD WORLD DATA
# -----------------------------------------------------------------------------

final_df = pd.concat(
    [df_unpivot, df_world],
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
                StructField("Indicator_Code", StringType(), False),
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
                                    add_months(max("Date"),3).alias("max_date")
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

filled_df = filled_df.withColumn(
    "Indicator_Code",
    last("Indicator_Code", ignorenulls=True).over(
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

filled_df.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.bronze.DRV_Oxford")

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
