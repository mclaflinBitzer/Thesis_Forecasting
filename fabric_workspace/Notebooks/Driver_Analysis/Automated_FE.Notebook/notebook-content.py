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


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#original drivers
compiled_drivers = spark.read.table("Sales_Forecasting.silver.compiled_drivers").select("Country","Indicator","Region","Date","Value")

# creating aggregated by region versions
aggregated_feature_set = compiled_drivers.groupBy("Region","Indicator","Date").agg(sum("Value").alias("Value"))
aggregated_feature_set = aggregated_feature_set.withColumn("Country", col("Region"))

# remove any drivers from the original set that aren't at "World" aggregation
compiled_drivers = compiled_drivers.filter(col("Region")=="World")

# join filtered original set w/ the aggregated drivers by region
compiled_drivers = compiled_drivers.unionByName(aggregated_feature_set)

compiled_drivers = compiled_drivers.withColumn('feature_serie', concat_ws("__","Country","Indicator","Region")).withColumnRenamed('Value','feature_value')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_pairs = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.topline_pairs")
topline_pairs = topline_pairs.withColumn('feature_serie', concat_ws("__",'Country',"Indicator","feature_region"))

topline_data = spark.read.table("Sales_Forecasting.silver.topline_cutoff_data").withColumnRenamed('Quantity','target_value')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_joined_data = topline_data.join(broadcast(topline_pairs), 'series','inner')
topline_joined_data = topline_joined_data.join(compiled_drivers, ['feature_serie','Date'],'inner')
display(topline_joined_data.limit(30))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def tsfresh_extraction(df):

    feature_serie = df['feature_serie'][0]

    df = df.sort_values(['Date'])
    X = extract_features(
        df, 
        column_id='feature_serie',
        column_sort='Date',
        column_value='feature_value')
        
    impute(X)

    X = X.reset_index().melt(
        id_vars='index',
        var_name='tsfresh_feature',
        value_name='value'
    )


    X.rename(columns={'index':'feature_serie'}, inplace=True)

    X['series'] = serie

    return X[['series','feature_serie','tsfresh_feature','value']]
   


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(test.limit(3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

tsfresh_schema = StructType([
    StructField('feature_serie',StringType(), False),
    StructField('tsfresh_feature', StringType(), False),
    StructField('value', DoubleType(), False)
])

tsfresh_feature_output = compiled_drivers.groupBy('feature_serie').applyInPandas(tsfesh_extraction, schema=tsfresh_schema)
display(tsfresh_feature_output)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

test = topline_joined_data.filter(col('series')=="ALU").select('series','feature_serie','Date','feature_value')
test_schema = StructType([
    StructField("series",StringType(), False),
    StructField("feature_series", StringType(), False),
    StructField('value', DoubleType(), False)
])
output = test.groupBy('series').applyInPandas(tsfresh_extraction, schema=test_schema)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

compiled_drivers.columns

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(output)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(output.count())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

temp = topline_joined_data.filter(col('series')=="ALU").select('series','feature_serie','Date','feature_value')

df = temp.toPandas()
display(df)

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
