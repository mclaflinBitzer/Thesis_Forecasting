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

%pip install tsfresh

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *
import pandas as pd
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

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

col_renamed = {"Quantity":"target_value","Value":"feature_value"}

## MIDDLE
M_series = ['series']
M_DRV_GRP_COLS = ['Region','Indicator']
M_ACT_GRP_COLS = ['Product_Category', 'Region','series']




# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#original drivers
compiled_drivers = spark.read.table("Sales_Forecasting.silver.compiled_drivers").select("Country","Indicator","Region","Date","Value")

print(compiled_drivers.columns)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

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

# CELL ********************

T_data = spark.read.table("Sales_Forecasting.silver.topline_cutoff_data")
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
T_joined = T_joined.withColumnsRenamed(col_renamed)
display(T_joined.limit(3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
