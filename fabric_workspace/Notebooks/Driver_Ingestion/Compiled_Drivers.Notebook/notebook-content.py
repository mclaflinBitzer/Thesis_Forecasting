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

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Extracting all tables within the lakehouse

# CELL ********************

lakehouse = "Sales_Forecasting.bronze"
tables = spark.catalog.listTables(lakehouse)
tables

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Extracting Driver tables and unioning into one dataframe

# CELL ********************

driver_tables= []
for t in tables:
    if t.name.startswith("drv"):
        print(f"{t.name} is a driver table & has been added to the driver_table list")
        driver_tables.append(t.name)
    else:
        print(f"{t.name} is not a driver table and has been skipped")


dfs = []
print("reading tables")

for name in driver_tables:
    print(f"Reading {name}")
    df = spark.read.table(lakehouse+"."+name).select("Country", "Indicator", "Date", "Value")
    df = df.withColumn("source_table", lit(name))
    dfs.append(df)

if dfs:
    joined_df = dfs[0]
    for df in dfs[1:]:
        joined_df = joined_df.unionByName(df, allowMissingColumns=True)
else:
    joined_df = None


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(joined_df.limit(5))
display(joined_df.select("source_table").distinct())
display(joined_df.count())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

joined_df.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.silver.Compiled_Drivers")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
