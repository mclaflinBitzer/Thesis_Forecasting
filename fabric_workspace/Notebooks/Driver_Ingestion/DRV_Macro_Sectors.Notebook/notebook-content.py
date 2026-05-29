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
# FILE PATH (Fabric Lakehouse)
# -----------------------------------------------------------------------------

file_path = (
    "/lakehouse/default/Files/Driver_Data/"
    "Macroeconomic_Sectors_Predictive_GlobalData.xlsx"
)

# -----------------------------------------------------------------------------
# READ EXCEL SHEET
# -----------------------------------------------------------------------------

df_read = pd.read_excel(
    file_path,
    sheet_name="Report",
    header=None
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************




# -----------------------------------------------------------------------------
# REMOVE TOP 14 ROWS
# -----------------------------------------------------------------------------

df_raw = df_read.iloc[16:].reset_index(drop=True)


# -----------------------------------------------------------------------------
# PROMOTE HEADERS
# -----------------------------------------------------------------------------

df_raw.columns = df_raw.iloc[0]

df_raw = df_raw.iloc[1:].reset_index(drop=True)


# -----------------------------------------------------------------------------
# RENAME FIRST COLUMNS
# -----------------------------------------------------------------------------
df_raw.columns = ["Country", "Indicator", "Unit"] + list(df_raw.columns[3:])


# -----------------------------------------------------------------------------
# REMOVE FIRST DATA ROW (as per Power Query)
# -----------------------------------------------------------------------------

df_raw = df_raw.iloc[1:].reset_index(drop=True)


# -----------------------------------------------------------------------------
# FILL DOWN COUNTRY
# -----------------------------------------------------------------------------

df_raw["Country"] = df_raw["Country"].replace("", np.nan).ffill()

## replace "-" with nan
df_raw = df_raw.replace("-", np.nan)

# -----------------------------------------------------------------------------
# CLEAN COLUMN NAMES
# -----------------------------------------------------------------------------

df_raw.columns = [
    str(col).replace(".0", "").strip()
    for col in df_raw.columns
]

# -----------------------------------------------------------------------------
# IDENTIFY YEAR COLUMNS
# -----------------------------------------------------------------------------

year_cols = [
    col for col in df_raw.columns
    if str(col).isdigit()
]

# -----------------------------------------------------------------------------
# UNPIVOT (MELT)
# -----------------------------------------------------------------------------

df_unpivot = df_raw.melt(
    id_vars=["Country", "Indicator", "Unit"],
    value_vars=year_cols,
    var_name="Year",
    value_name="Value"
)


# -----------------------------------------------------------------------------
# CLEAN VALUE COLUMN
# -----------------------------------------------------------------------------

df_unpivot["Value"] = df_unpivot["Value"].replace("-", np.nan)

df_unpivot["Value"] = pd.to_numeric(
    df_unpivot["Value"],
    errors="coerce"
)


# -----------------------------------------------------------------------------
# CONVERT YEAR → DATE
# -----------------------------------------------------------------------------

df_unpivot["Date"] = pd.to_datetime(
    df_unpivot["Year"].astype(str) + "-01-01"
)

df_unpivot = df_unpivot.drop(columns=["Year"])

# -----------------------------------------------------------------------------
# FINAL COLUMN ORDER
# -----------------------------------------------------------------------------

df_unpivot = df_unpivot[
    ["Country", "Indicator", "Unit", "Date", "Value"]
]

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

df_world = df_unpivot.groupby(["Indicator", "Unit", "Date"], as_index=False)["Value"].sum()
df_world["Country"] = "World"

df_world = df_world[["Country", "Indicator", "Unit", "Date", "Value"]]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# -----------------------------------------------------------------------------
# APPEND WORLD DATAFRAME
# (assumes Macro_Sector_GD_World already exists)
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

spark_schema = StructType([
            StructField("Country", StringType(), False),
            StructField("Indicator", StringType(), False),
            StructField("Unit", StringType(), True),
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

filled_df = filled_df.withColumn(
    "Unit",
    last("Unit", ignorenulls=True).over(
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

filled_df.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.bronze.DRV_Macro_Sectors")

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
