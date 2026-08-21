# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "warehouse": {
# META       "default_warehouse": "ddfdd8f1-f4f6-4bf2-acda-546b8684dd80",
# META       "known_warehouses": [
# META         {
# META           "id": "ddfdd8f1-f4f6-4bf2-acda-546b8684dd80",
# META           "type": "Lakewarehouse"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

import pandas as pd
from pyspark.sql.functions import *

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# PARAMETERS CELL ********************

Topline = True
driver_status = 'No_Drivers'

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

%run Config_Prod

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

config_obj = build_config(
    topline=Topline,
    driver_status=driver_status
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

act_table_name = config_obj['init_act_table']
cutoff_data_table = config_obj['cutoff_data_table']
cutoff_data_table

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## creating dataframe w/ the cutoff dates
data = []
for key, item in HISTORICAL_CUTOFF_DATES.items():
    data.append({
                "series":key,
                "cutoff_date":item
                })
df = pd.DataFrame(data)
cutoff_dates = spark.createDataFrame(df)
cutoff_dates = cutoff_dates.withColumn("cutoff_date", to_date(col("cutoff_date"), "dd-MM-yyyy"))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

act_data = spark.read.table(act_table_name)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

filtered_data = (
    act_data.alias("d")
    .join(
        cutoff_dates.alias("c"),
        col("d.series").startswith(col("c.series")),
        "inner"
    )
    .filter(col("d.Date")>=col("c.historical_cutoff_date"))
    .select("d.*")
)
display(filtered_data.limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
