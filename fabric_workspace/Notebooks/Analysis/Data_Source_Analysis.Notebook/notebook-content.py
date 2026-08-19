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

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

T_Arimax_output = spark.read.parquet("abfss://991f5e4b-c174-4ff2-992e-feb17d49d25a@onelake.dfs.fabric.microsoft.com/22746de3-183e-4327-a844-dceda0b7165c/Files/Forecasting/Topline/No_Drivers/Arimax_Output.parquet")
T_Arimax_output.cache()
T_actuals = spark.read.table("Sales_Forecasting.silver.topline_cutoff_data")
T_actuals.cache()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(T_Arimax_output.select('Product_Category').distinct())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print(T_actuals.columns)
print(T_Arimax_output.columns)

test = (
    T_actuals
    .withColumn("Product_Category", concat(col("Product_Category"),lit("__ACT")))
    .withColumnRenamed("Quantity","Value")
    .select("Product_Category","Date","Value")
    .unionByName(
        T_Arimax_output
        .groupBy("Product_Category","Date").agg(avg("Forecast").alias("Value"))
        .withColumn("Product_Category", concat(col("Product_Category"), lit("__FORECAST")))
        .select("Product_Category","Date","Value")
    )
)
display(test.filter(col('Product_Category').like('MAERSK_ELECTRONICS%')))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(T_actuals)
display(
    T_Arimax_output.groupBy('Product_Category','Date')
    .agg(avg('Forecast').alias('Forecast'))
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import *

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

data = spark.read.table("Sales_Forecasting.bronze.filtered_data")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

data.printSchema()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

middle = data.groupBy('Date','Product_Category').agg(sum('Quantity').alias('Quantity'))
topline = data.groupBy('Date','Product_Category', 'Region').agg(sum('Quantity').alias('Quantity')) 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

middle_pdf = middle.toPandas()
middle_pdf.info()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_pdf = topline.toPandas()
topline_pdf.info()
count_zero = (topline.filter(
    (col('Quantity')==0) | (col('Quantity').isNull())
    ).count()
)
total_count = topline.count()
print(count_zero)
print(total_count)
print(f"percentage zero {count_zero/total_count}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

raw_data = spark.read.table('Sales_Forecasting.bronze.Raw_Analyse_Sales_BPC')

display(raw_data)

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
