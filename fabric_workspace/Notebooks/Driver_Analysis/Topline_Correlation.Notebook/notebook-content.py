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

from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.functions import col
from pyspark.sql.window import Window

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Configurations

# CELL ********************

LAG_MONTHS = [3, 6, 9, 12, 15, 18, 21, 24]
LEAD_MONTHS = [3, 6, 9, 12]

FEATURE_COLS = (
    ["Value"] +
    [f"lag_{n}" for n in LAG_MONTHS] +
    [f"lead_{n}" for n in LEAD_MONTHS]
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Methods

# MARKDOWN ********************

# ##### Creating lag features

# CELL ********************

def lag_creation(df, lag_months):

    ## creating the window partitioned on "Indicator" and ordered by Date
    w_lag = (
        Window.partitionBy("Indicator").orderBy("Date")
    )

    ## Creation of the lag columns w/ lagged values 
    for lag_n in lag_months:
        df = df.withColumn(f"lag_{lag_n}", lag("Value", lag_n).over(w_lag))

    return df


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ##### Creating lead features

# CELL ********************

def lead_creation(df, lead_months):

    ## creating the window partitioned on "Indicator" and ordered by Date
    w_lead = Window.partitionBy("Indicator").orderBy("Date")

    ## Creation of the lead columns w/ leaded values
    for lead_n in lead_months:
        df = df.withColumn(f"lead_{lead_n}", lead("Value", lead_n).over(w_lead))

    return df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ##### Join historical data w/ drivers/generated features

# CELL ********************

def joining_features_historical(driver_df, historical_df, feature_cols):
    
    ## selecting only the needed columns
    features_df = driver_df.drop("source_table")


    ## Joining drivers to the target series
    model_df = (
        historical_df
        .join(
            features_df,
            on=["Date"],
            how="left"
        )
    )


    stack_expr = ", ".join([f"'{c}', {c}" for c in feature_cols])

    long_df = model_df.selectExpr("series", "Indicator", "Date", "Quantity",
                                f"""
                                stack(
                                    {len(feature_cols)},
                                    {stack_expr}
                                    ) as (feature_name, feature_value)
                                """)

    return long_df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ##### Correlation Analysis

# CELL ********************

def corr_analysis(df):

    ## creating the correlation dataframe
    corr_df = (
        df.groupBy("series", "Indicator", "feature_name")\
                .agg(corr("Quantity","feature_value").alias("Correlation"))
    )

    ## ranking the drivers based on the absolute correlation values
    corr_df = corr_df.withColumn("abs_corr", abs(col("correlation")))

    w = Window.partitionBy("series").orderBy(desc("abs_corr"))

    top_drivers = corr_df.withColumn("rank", row_number().over(w)).filter(col("rank") <= 20)

    return corr_df, top_drivers

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Implementation

# MARKDOWN ********************

# ### Ingesting Data

# CELL ********************

raw_drivers = spark.read.table("Sales_Forecasting.silver.compiled_drivers")
topline = spark.read.table("Sales_Forecasting.bronze.topline_data")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Creating lagged features

# CELL ********************

## creating a unique indicator column that contains "country" as well for filtering / differentiation during correlation analysis
raw_drivers = raw_drivers.withColumn("Unique_Indicator", concat_ws("__", col("Country"), col("Indicator")))
raw_drivers = raw_drivers.drop("Country", "Indicator").withColumnRenamed("Unique_Indicator","Indicator")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

oxford_world_drivers = raw_drivers.filter(
                                            (col("source_table") == "drv_oxford") & 
                                            (col("Indicator").contains("World"))
                                        )

oxford_country_drivers = raw_drivers.filter(
                                            (col("source_table") == "drv_oxford") &
                                            (~col("Indicator").contains("World"))
                                        )

gd_world_drivers = raw_drivers.filter(
                                        (col("source_table") != "drv_oxford") &
                                        (col("Indicator").contains("World"))
                                        )
gd_country_drivers = raw_drivers.filter(
                                            (col("source_table") != "drv_oxford") &
                                            (~col("Indicator").contains("World"))
                                        )

all_world_drivers = raw_drivers.filter(col("Indicator").contains("World"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## creating the lag and lead columns/values per driver
oxford_world_drivers = lag_creation(oxford_world_drivers, LAG_MONTHS)
oxford_world_drivers = lead_creation(oxford_world_drivers, LEAD_MONTHS)

oxford_country_drivers = lag_creation(oxford_country_drivers, LAG_MONTHS)
oxford_country_drivers = lead_creation(oxford_country_drivers, LEAD_MONTHS)

gd_world_drivers = lag_creation(gd_world_drivers, LAG_MONTHS)
gd_world_drivers = lead_creation(gd_world_drivers, LEAD_MONTHS)

gd_country_drivers = lag_creation(gd_country_drivers, LAG_MONTHS)
gd_country_drivers = lead_creation(gd_country_drivers, LEAD_MONTHS)

## all world drivers
all_world_drivers = lag_creation(all_world_drivers, LAG_MONTHS)
all_world_drivers = lead_creation(all_world_drivers, LEAD_MONTHS)

## all driver info as well
all_drivers = lag_creation(raw_drivers, LAG_MONTHS)
all_drivers = lead_creation(all_drivers, LEAD_MONTHS)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## joining w/ historical data
oxford_world_joined = joining_features_historical(oxford_world_drivers, topline, FEATURE_COLS)
oxford_country_joined = joining_features_historical(oxford_country_drivers, topline, FEATURE_COLS)


gd_world_joined = joining_features_historical(gd_world_drivers, topline, FEATURE_COLS)
gd_country_joined = joining_features_historical(gd_country_drivers, topline, FEATURE_COLS)

all_world_drivers_joined = joining_features_historical(all_world_drivers, topline, FEATURE_COLS)
all_drivers_joined = joining_features_historical(all_drivers, topline, FEATURE_COLS)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## Correlation Analysis
oxford_world_corr, oxford_world_top_corr = corr_analysis(oxford_world_joined)
oxford_country_corr, oxford_country_top_corr = corr_analysis(oxford_country_joined)

gd_world_corr, gd_world_top_corr = corr_analysis(gd_world_joined)
gd_country_corr, gd_country_top_corr = corr_analysis(gd_country_joined)

all_world_corr, all_world_top_corr = corr_analysis(all_world_drivers_joined)
all_corr, all_top_corr = corr_analysis(all_drivers_joined)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def ordered_df(df):
    df = df.orderBy("series",desc("abs_corr"))
    return df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

oxford_world_corr = ordered_df(oxford_world_corr)
oxford_country_corr = ordered_df(oxford_country_corr)

gd_world_corr = ordered_df(gd_world_corr)
gd_country_corr = ordered_df(gd_country_corr)

all_world_corr = ordered_df(all_world_corr)
all_corr = ordered_df(all_corr)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(oxford_world_top_corr.filter(col("rank")<=5))
display(oxford_country_top_corr.filter(col("rank")<=5))
display(all_world_top_corr.filter(col("rank")<=5))
display(all_top_corr.filter(col("rank")<=5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(oxford_world_top_corr.groupBy("Indicator").agg(count("*").alias("Count")).orderBy(desc(col("Count"))))
display(oxford_country_top_corr.groupBy("Indicator").agg(count("*").alias("Count")).orderBy(desc(col("Count"))))
display(all_world_top_corr.groupBy("Indicator").agg(count("*").alias("Count")).orderBy(desc(col("Count"))))
display(all_top_corr.groupBy("Indicator").agg(count("*").alias("Count")).orderBy(desc(col("Count"))))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Exporting / saving the outputs

# CELL ********************

def writing_files(df,name):
    pd_df = df.toPandas()
    local_path = f"/tmp/{name}.xlsx"
    pd_df.to_excel(local_path)

    dest_path = f"/lakehouse/default/Files/Driver_Data/Driver_Analysis/{name}.xlsx"
    shutil.copy(local_path, dest_path)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import shutil

writing_files(oxford_world_corr, "Oxford_World_Correlation")
writing_files(oxford_world_top_corr, "Oxford_World_Top_Correlation")
writing_files(oxford_country_corr, "Oxford_Country_Correlation")
writing_files(oxford_country_top_corr, "Oxford_Country_Top_Correlation")

writing_files(gd_world_corr, "GD_World_Correlation")
writing_files(gd_world_top_corr, "GD_World_Top_Correlation")
writing_files(gd_country_corr, "GD_Country_Correlation")
writing_files(gd_country_top_corr, "GD_Country_Top_Correlation")

writing_files(all_world_corr, "All_Driver_World_Correlation")
writing_files(all_world_top_corr, "All_Drivers_World_Top_Correlation")

writing_files(all_corr, "All_Driver_Correlation")
writing_files(all_top_corr, "All_Driver_Top_Correlation")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
