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
from pyspark.sql.types import *
import pandas as pd

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## TOPLINE
T_series = ['series']
T_DRV_GRP_COLS = ['Indicator']
T_ACT_GRP_COLS = ['Product_Category','series']

T_cols = T_ACT_GRP_COLS + T_DRV_GRP_COLS

T_actuals_table = "Sales_Forecasting.silver.topline_cutoff_data"

## MIDDLE
M_series = ['series']
M_DRV_GRP_COLS = ['Region','Indicator']
M_ACT_GRP_COLS = ['Product_Category', 'Region','series']

M_actuals_table = "Sales_Forecasting.silver.middle_cutoff_data"


## shared
col_renamed = {"Quantity":"target_value","Value":"feature_value"}
driver_table = "Sales_Forecasting.silver.compiled_drivers"


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Reading Data

# CELL ********************

#original drivers
compiled_drivers = spark.read.table(driver_table).select("Country","Indicator","Region","Date","Value")

# actual data
T_data = spark.read.table(T_actuals_table)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Driver Processing

# CELL ********************

aggregated_drivers = compiled_drivers.groupBy(*T_DRV_GRP_COLS,'Date').agg(sum('Value').alias('Value'))

if 'Indicator' in T_DRV_GRP_COLS and len(T_DRV_GRP_COLS) > 1:
    world_agg = compiled_drivers.groupBy('Indicator','Date').agg(sum("Value").alias("Value"))
    
    for cols in T_DRV_GRP_COLS:
        if cols!='Indicator':
            world_agg = world_agg.withColumn(cols, lit("World"))
            print(f"{cols} added using .withColumn, populated with lit(World)")
    aggregated_drivers = aggregated_drivers.unionByName(world_agg)
else:
    print('world agg already done')


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Feature Engineering / Expansion

# CELL ********************

def feature_engineering(df, drv_grp_cols):
    ## Feature Transformations to do
        ## levels
        ## Rolling Mean / STD (3,6,12)
        ## Growth: YoY, MoM, diff

    ## levels
    df = df.withColumn("level", col("Value")).drop("Value")

    ## Window creation
    w = Window.partitionBy(*drv_grp_cols).orderBy("Date")
    w_3 = Window.partitionBy(*drv_grp_cols).orderBy("Date").rowsBetween(-2,0)
    w_6 = Window.partitionBy(*drv_grp_cols).orderBy("Date").rowsBetween(-5,0)
    w_12 = Window.partitionBy(*drv_grp_cols).orderBy("Date").rowsBetween(-11,0)
    w_18 = Window.partitionBy(*drv_grp_cols).orderBy("Date").rowsBetween(-17,0)
    w_24 = Window.partitionBy(*drv_grp_cols).orderBy("Date").rowsBetween(-23,0)

    windows = {
        3: w_3,
        6: w_6,
        12: w_12,
        18: w_18,
        24: w_24
    }

    

    ## Rolling Mean & Stddev
    for size, win in windows.items():
        df = df.withColumn(f"rolling_mean_{size}", avg("level").over(win))
        df = df.withColumn(f"rolling_std_{size}", stddev("level").over(win))


    # df = df.withColumn("rolling_mean_3", avg("level").over(w_3))
    # df = df.withColumn("rolling_mean_6", avg("level").over(w_6))
    # df = df.withColumn("rolling_mean_12", avg("level").over(w_12))
    # df = df.withColumn("rollwing_mean_18", avg("level").over(w_18))
    # df = df.withColumn("rolling_mean_24", avg("level").over(w_24))

    # df = df.withColumn("rolling_std_3", stddev("level").over(w_3))
    # df = df.withColumn("rolling_std_6", stddev("level").over(w_6))
    # df = df.withColumn("rolling_std_12", stddev("level").over(w_12))
    # df = df.withColumn("rolling_std_18", stddev("level").over(w_18))
    # df = df.withColumn("rolling_std_24", stddev("level").over(w_24))


    ## Growth: YoY, MoM, Diff
    df = df.withColumn("diff", (col("level") - lag("level", 1).over(w)))
    df = df.withColumn("YoY_pct", 
                            when(lag("level",12).over(w) == 0, None)
                            .otherwise((col("level")/lag("level",12).over(w)) -1))
    df = df.withColumn("MoM_pct",
                            when(lag("level",1).over(w)==0, None)
                            .otherwise((col("level") / lag("level", 1).over(w))-1))

    return df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

expanded_features = feature_engineering(aggregated_drivers, T_DRV_GRP_COLS)
display(expanded_features.limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Classical Feature Processing
# - removal of constant features
# - ccf w/ lag identification
# - corr clustering removal
# - Elastic Net selection
#         
#         - scale data for elastic net
#         - select features based output per target series
# 
# - Finalize prior to Model ingestion

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


T_data_distinct = T_data.select(*T_ACT_GRP_COLS).distinct()
T_drv_distinct = aggregated_drivers.select(*T_DRV_GRP_COLS).distinct()
T_drv_distinct = T_drv_distinct.withColumn('feature_serie', concat_ws("__", *T_DRV_GRP_COLS))

join_col = []
for a_col in T_ACT_GRP_COLS:
    for d_col in T_DRV_GRP_COLS:
        if d_col == a_col:
            join_col.append(d_col)
            print(f"{d_col} added to join col list")

if len(join_col) == 0:
    print(f"no shared columns so a cross join was done")
    T_pairs = T_data_distinct.crossJoin(T_drv_distinct)
else:
    print(f"shared columns so the join was on {join_col}")
    T_pairs = T_data_distinct.join(T_drv_distinct, join_col, 'inner')


T_data = T_data.join(broadcast(T_pairs), T_ACT_GRP_COLS, 'inner')
T_aggregated_drivers = aggregated_drivers.join(broadcast(T_pairs), T_DRV_GRP_COLS, 'inner')
T_aggregated_drivers = T_aggregated_drivers.withColumnRenamed("Value","feature_value")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

T_joined = T_data.join(T_aggregated_drivers, [*T_cols,'Date'], 'inner')


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
