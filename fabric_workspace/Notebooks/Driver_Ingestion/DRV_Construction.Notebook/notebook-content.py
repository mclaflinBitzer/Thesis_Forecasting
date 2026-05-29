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



path = "abfss://991f5e4b-c174-4ff2-992e-feb17d49d25a@onelake.dfs.fabric.microsoft.com/22746de3-183e-4327-a844-dceda0b7165c/Files/Driver_Data/Construction_Sectors_Predictive_GlobalData.xlsx"

pdf = pd.read_excel(path, sheet_name="Report", header=None)

# skipping the first 16 rows
pdf = pdf.iloc[18:]

## Promoting to headers
pdf.columns = pdf.iloc[0]
pdf = pdf[1:]

# rename Col1 to Country
pdf.columns.values[0] = "Country"
## skip one row
pdf = pdf.iloc[1:]


# add indicator column
pdf["Indicator"] = "Construction_Sector"

#ordered columns
ordered_cols = (
    ["Country", "Indicator"] +
    [col for col in pdf.columns if col not in ["Country", "Indicator"]]
)

pdf = pdf[ordered_cols]

# extract the year columns
year_cols = [col for col in pdf.columns if str(col).isdigit()]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_unpivot = pdf.melt(id_vars=["Country", "Indicator"], value_vars=year_cols, var_name="Date", value_name="Value")

# CONVERT TYPES

df_unpivot["Date"] = pd.to_datetime(
    df_unpivot["Date"].astype(str) + "-01-01"
)

df_unpivot["Value"] = pd.to_numeric(
    df_unpivot["Value"],
    errors="coerce"
)

# -----------------------------------------------------------------------------
# GROUP BY INDICATOR + DATE
# -----------------------------------------------------------------------------

df_grouped = (
    df_unpivot
    .groupby(
        ["Country","Indicator", "Date"],
        as_index=False
    )["Value"]
    .sum()
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# create "world" grouping
world_df = df_grouped.groupby(["Indicator","Date"], as_index=False)["Value"].sum()

world_df["Country"] = "World"
world_df = world_df[["Country","Indicator","Date","Value"]]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# -----------------------------------------------------------------------------
# APPEND / CONCATENATE
# -----------------------------------------------------------------------------

final_df = pd.concat(
    [df_grouped, world_df],
    ignore_index=True
)



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
    StructField("Value", DoubleType(), False)
])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## Convert to spark dataframe
spark_df = spark.createDataFrame(final_df, schema=spark_schema)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

driver_time_bounds = (
                        spark_df
                                .filter(col("Value").isNotNull())
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

filled_df.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.bronze.DRV_Construction_Sector")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
