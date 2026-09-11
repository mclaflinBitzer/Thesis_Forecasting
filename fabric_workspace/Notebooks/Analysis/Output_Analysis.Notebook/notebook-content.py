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
from functools import reduce
from pyspark.sql.types import *
import pandas as pd
from pyspark.sql.window import Window

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## Parameters
Topline = True

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## BASELINE output directories
base_parquet_dir = "abfss://991f5e4b-c174-4ff2-992e-feb17d49d25a@onelake.dfs.fabric.microsoft.com/22746de3-183e-4327-a844-dceda0b7165c/Files/Forecasting"
base_output_dir = "abfss://991f5e4b-c174-4ff2-992e-feb17d49d25a@onelake.dfs.fabric.microsoft.com/22746de3-183e-4327-a844-dceda0b7165c/Files/Eval"

forecast_sub_folders = ['No_Drivers','Automated_Drivers','Manual_Drivers']

if Topline:

    actuals_table = "Sales_Forecasting.silver.topline_cutoff_data"
    initial_target_col = "Quantity"

    DRV_GRP_COLS = ['Indicator']
    ACT_GRP_COLS = ['Product_Category','series']

    ## OUTPUT DIRECTORIES
    parquet_dir = base_parquet_dir + "/Topline/"
    all_forecast_dirs = []

    seasonal_baseline_dir = parquet_dir + "Seasonal_Baseline_Output.parquet"
    all_forecast_dirs.append(seasonal_baseline_dir)

    Holts_dir = parquet_dir + "No_Drivers/Holts_Output.parquet"
    all_forecast_dirs.append(Holts_dir)

    for folder in forecast_sub_folders:
        all_forecast_dirs.append((parquet_dir + folder + "/Arimax_Output.parquet"))         # ARIMAX DIR
        all_forecast_dirs.append((parquet_dir + folder + "/Sarimax_output.parquet"))        # SARIMAX DIR
        all_forecast_dirs.append((parquet_dir + folder + "/Prophet_output.parquet"))        # PROPHET DIR  
        all_forecast_dirs.append((parquet_dir + folder + "/XGBoost_Output.parquet"))        # XGBOOST DIR
        all_forecast_dirs.append((parquet_dir + folder + "/TFT_Forecast_Output.parquet"))    # TFT DIR
    
    output_dir = base_output_dir + "/Topline/"
    output_table = 'Sales_Forecasting.Topline_Eval.'


else:

    actuals_table = "Sales_Forecasting.silver.middle_cutoff_data"
    initial_target_col = 'Quantity'

    DRV_GRP_COLS = ['Region','Indicator']
    ACT_GRP_COLS = ['Product_Category', 'Region','series']


    ## OUTPUT DIRECTORIES
    parquet_dir = base_parquet_dir + "/Middle/"
    all_forecast_dirs = []

    seasonal_baseline_dir = parquet_dir + "Seasonal_Baseline_Output.parquet"
    all_forecast_dirs.append(seasonal_baseline_dir)

    Holts_dir = parquet_dir + "No_Drivers/Holts_Output.parquet"
    all_forecast_dirs.append(Holts_dir)

    for folder in forecast_sub_folders:
        all_forecast_dirs.append((parquet_dir + folder + "/Arimax_Output.parquet"))         # ARIMAX DIR
        all_forecast_dirs.append((parquet_dir + folder + "/Sarimax_output.parquet"))        # SARIMAX DIR
        all_forecast_dirs.append((parquet_dir + folder + "/Prophet_output.parquet"))        # PROPHET DIR  
        all_forecast_dirs.append((parquet_dir + folder + "/XGBoost_Output.parquet"))        # XGBOOST DIR
        all_forecast_dirs.append((parquet_dir + folder + "/TFT_Forecast_Output.parquet"))    # TFT DIR
    

    output_dir = base_output_dir + "/Middle/"
    output_table = 'Sales_Forecasting.Middle_Eval.'



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

all_forecast_dfs = []
for file_dir in all_forecast_dirs:
    print(file_dir)
    temp_file = spark.read.parquet(file_dir).select(*ACT_GRP_COLS, 'Training_End_Date','Date','Forecast_Horizon','Forecast','Forecaster','Drivers_Used_Flag')
    all_forecast_dfs.append(temp_file)

unioned_dfs = reduce(
    lambda df1, df2: df1.unionByName(df2),
    all_forecast_dfs
)
unioned_dfs.cache()

display(unioned_dfs)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

actuals_df = spark.read.table(actuals_table)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Creating shared Forecast table w/ baseline and actuals & key table

# CELL ********************

actuals_prep = (
    actuals_df
    .withColumnRenamed('Quantity','Forecast')
    .withColumn('Forecaster', lit('ACTUALS'))
    .withColumn('Drivers_Used_Flag', lit('ACTUALS_NO_DRIVERS'))
)
compiled_values = unioned_dfs.unionByName(actuals_prep, allowMissingColumns=True)


compiled_values = compiled_values.withColumn('identifier_col', concat_ws('___', *ACT_GRP_COLS, 'Forecaster','Drivers_Used_Flag'))

key_table = compiled_values.select('identifier_col',*ACT_GRP_COLS, 'Forecaster', 'Drivers_Used_Flag').distinct()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## creating filtered baseline / unioned_df for metrics

# CELL ********************

baseline = unioned_dfs.filter(col('Forecaster')=='Seasonal_Baseline').withColumn('identifier_col', concat_ws('___', *ACT_GRP_COLS, 'Forecaster','Drivers_Used_Flag'))


unioned_filtered_dfs = unioned_dfs.filter(col('Forecaster')!='Seasonal_Baseline').withColumn('identifier_col', concat_ws('___', *ACT_GRP_COLS, 'Forecaster','Drivers_Used_Flag'))


actuals_metric_df = actuals_df.withColumnRenamed('Quantity','Actuals')


unioned_actuals_df = actuals_metric_df.join(
    unioned_filtered_dfs,
    [*ACT_GRP_COLS,'Date'],
    'inner'
)



baseline_actuals = baseline.join(
    actuals_metric_df,
    [*ACT_GRP_COLS,'Date'],
    'inner'
)


baseline_actuals = baseline_actuals.withColumn('Baseline_Abs_Error', abs(col('Actuals')-col('Forecast')))


eval_table = (
    unioned_actuals_df
    .withColumn('Forecast_Error', col('Actuals')-col('Forecast'))
    .withColumn('Forecast_Abs_Error', abs(col('Actuals')-col('Forecast')))
)


eval_table = eval_table.join(
    baseline_actuals.select(*ACT_GRP_COLS, 'Date','Training_End_Date', 'Forecast_Horizon','Baseline_Abs_Error'),
    [*ACT_GRP_COLS, 'Date','Training_End_Date','Forecast_Horizon'],
    'inner'
)



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Creating metric tables

# CELL ********************

def calculate_metrics(df, group_cols):
        
    return (
        df
        .groupBy(*group_cols)
        .agg(
            count("*").alias("N"),
            (avg("Forecast_Error")*-1).alias("ME"),
            avg("Forecast_Abs_Error").alias("MAE"),
            sum("Forecast_Abs_Error").alias("Forecast_Abs_Error_Sum"),
            sum(abs(col("Actuals"))).alias("Actual_Sum"),
            sum("Baseline_Abs_Error").alias("Baseline_Abs_Error_Sum")
        )
        .withColumn(
            "WAPE",
            when(
                col("Actual_Sum") != 0,
                col("Forecast_Abs_Error_Sum") /
                col("Actual_Sum")
            )
        )
        .withColumn(
            "MASE",
            when(
                col("Baseline_Abs_Error_Sum") != 0,
                col("Forecast_Abs_Error_Sum") /
                col("Baseline_Abs_Error_Sum")
            )
        )
        .drop(
            "Forecast_Abs_Error_Sum",
            "Actual_Sum",
            "Baseline_Abs_Error_Sum"
        )
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Dynamically setting schema

# CELL ********************

def cast_schema(df):
    ## ID cols are the grouping / identifier cols
    """
    Rules:
        - Date, Training_End_Date -> DateType
        - Metrics: [Forecast, Actuals, Forecast_Error, Forecast_Abs_Error, Baseline_Abs_Error,
                    N, ME, MAE, WAPE, MASE] -> DoubleType
        - id_cols -> StringType
        - Forecast_Horizon -> IntegerType

    """

    double_metrics = ['Forecast', 'Actuals', 'Forecast_Error', 'Forecast_Abs_Error', 'Baseline_Abs_Error',
                    'N', 'ME', 'MAE', 'WAPE', 'MASE']

    int_metrics = ['N', 'Forecast_Horizon']
    
    select_exprs = []
    for field in df.schema.fields:
        name = field.name

        if (name == 'Date') | (name == 'Training_End_Date'):
            select_exprs.append(col(name).cast(DateType()).alias(name))

        elif name in int_metrics:
            select_exprs.append(col(name).cast(IntegerType()).alias(name))

        elif name in double_metrics:
            select_exprs.append(col(name).cast(DoubleType()).alias(name))

        else:
            select_exprs.append(col(name).cast(StringType()))

    return df.select(*select_exprs)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Creating final dataframes / metrics

# CELL ********************

monthly_agg = compiled_values.groupBy(*ACT_GRP_COLS, 'Forecaster','Drivers_Used_Flag', 'Date').agg(avg('Forecast').alias('Forecast'))
monthly_agg = monthly_agg.withColumn('identifier_col', concat_ws("___", *ACT_GRP_COLS, col('Forecaster'),col('Drivers_Used_Flag')))
monthly_agg = cast_schema(monthly_agg)
monthly_agg.printSchema()


print('compiled_values')
compiled_values = cast_schema(compiled_values)
compiled_values.printSchema()

print('key_table')
key_table = cast_schema(key_table)
key_table.printSchema()

print('eval_table')
eval_table = cast_schema(eval_table)
eval_table.printSchema()

## Model level aggregation
print('model_metrics')
model_metrics = calculate_metrics(eval_table, group_cols=['Forecaster','Drivers_Used_Flag'])
model_metrics = model_metrics.withColumn('identifier_col', concat_ws("___", col('Forecaster'),col('Drivers_Used_Flag')))
model_metrics = cast_schema(model_metrics)
model_metrics.printSchema()

## Model x FH level aggregation
print('model_fh_metrics')
model_fh_metrics = calculate_metrics(eval_table, group_cols= ['Forecaster','Forecast_Horizon','Drivers_Used_Flag'])
model_fh_metrics = model_fh_metrics.withColumn('identifier_col', concat_ws("___", col('Forecaster'),col('Drivers_Used_Flag')))
model_fh_metrics = cast_schema(model_fh_metrics)
model_fh_metrics.printSchema()


## Model x Training End Date level aggregation
print('model_training_metrics')
model_training_metrics = calculate_metrics(eval_table, group_cols=['Forecaster','Training_End_Date','Drivers_Used_Flag'])
model_training_metrics = model_training_metrics.withColumn('identifier_col', concat_ws("___", col('Forecaster'),col('Drivers_Used_Flag')))
model_training_metrics = cast_schema(model_training_metrics)
model_training_metrics.printSchema()




## Model x series level aggregation
print('model_series_metrics')
model_series_metrics = calculate_metrics(eval_table, group_cols = [*ACT_GRP_COLS,'Forecaster','Drivers_Used_Flag'])
model_series_metrics = model_series_metrics.withColumn('identifier_col', concat_ws('___', *ACT_GRP_COLS, 'Forecaster','Drivers_Used_Flag'))
model_series_metrics = cast_schema(model_series_metrics)
model_series_metrics.printSchema()


## Model x series x FH metrics 
print('model series fh metrics')
model_series_fh_metrics = calculate_metrics(eval_table, group_cols= [*ACT_GRP_COLS,'Forecaster','Forecast_Horizon','Drivers_Used_Flag'])
model_series_fh_metrics = model_series_fh_metrics.withColumn('identifier_col', concat_ws('___', *ACT_GRP_COLS, 'Forecaster','Drivers_Used_Flag'))
model_series_fh_metrics = cast_schema(model_series_fh_metrics)
model_series_fh_metrics.printSchema()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Writing Output Files/Tables

# CELL ********************

## Writing output files

monthly_agg.write.mode('overwrite').saveAsTable(output_table+'monthly_agg')


print('compiled_values')
compiled_values.write.mode('overwrite').saveAsTable(output_table+'compiled_values')
compiled_values.write.mode('overwrite').parquet(output_dir+'compiled_values.parquet')

print('key_table')
key_table.write.mode('overwrite').saveAsTable(output_table+'key_table')
key_table.write.mode('overwrite').parquet(output_dir+'key_table.parquet')

print('eval_table')
eval_table.write.mode('overwrite').parquet(output_dir + "eval_table.parquet")
eval_table.write.mode('overwrite').saveAsTable(output_table+'eval_table')


## Model level aggregation

print('model_metrics')

model_metrics.write.mode('overwrite').parquet(output_dir + "model_metrics.parquet")
model_metrics.write.mode('overwrite').saveAsTable(output_table+'model_metrics')


## Model x FH level aggregation
print('model_fh_metrics')
model_fh_metrics.write.mode('overwrite').parquet(output_dir + "model_FH_metrics.parquet")
model_fh_metrics.write.mode('overwrite').saveAsTable(output_table+'model_fh_metrics')


## Model x Training End Date level aggregation
print('model_training_metrics')
model_training_metrics.write.mode('overwrite').parquet(output_dir + "model_training_metrics.parquet")
model_training_metrics.write.mode('overwrite').saveAsTable(output_table+'model_training_metrics')




## Model x series level aggregation
print('model_series_metrics')
model_series_metrics.write.mode('overwrite').parquet(output_dir + "model_series_metrics.parquet")
model_series_metrics.write.mode('overwrite').saveAsTable(output_table+'model_series_metrics')


## Model x series x FH metrics 
print('model series fh metrics')
model_series_fh_metrics.write.mode('overwrite').parquet(output_dir + "model_series_fh_metrics.parquet")
model_series_fh_metrics.write.mode('overwrite').saveAsTable(output_table+'model_series_fh_metrics')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## visual comparison

# CELL ********************

# actuals_vis_df = spark.read.table(actuals_table)
# unioned_vis_df = unioned_dfs.unionByName(actuals_vis_df, allowMissingColumns=True)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# prod_c = [
#     row["Product_Category"]
#     for row in unioned_vis_df
#         .select("Product_Category")
#         .distinct()
#         .collect()
# ]
# prod_c

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# for prod in prod_c:
#     display(
#         unioned_vis_df
#         .filter(
#             col('Product_Category')==prod
#         )
#         .groupBy('identifier_col','Date')
#         .agg(avg('Forecast').alias('Forecast'))
#         )


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


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Driver Impact Analysis  

# CELL ********************

base_drv_dir = 'abfss://991f5e4b-c174-4ff2-992e-feb17d49d25a@onelake.dfs.fabric.microsoft.com/22746de3-183e-4327-a844-dceda0b7165c/Files/Forecasting'
if Topline:
    base_drv_dir = base_drv_dir + '/Topline/'
else: 
    base_drv_dir = base_drv_dir + '/Middle/'  

automated_drv_dir = base_drv_dir + 'Automated_Drivers/'
manual_drv_dir = base_drv_dir + 'Manual_Drivers/'



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### ARIMAX


# CELL ********************

def arimax_drv(drv_dir):
    arimax_auto = spark.createDataFrame(pd.read_parquet(drv_dir + 'Arimax_Output.parquet'))
    max_training_date = arimax_auto.agg(max('Training_End_Date').alias('max_training_date')).collect()[0]['max_training_date']
    arimax_auto = arimax_auto.filter(col('Training_End_Date')==max_training_date)

    drivers_df = (
        arimax_auto
        .select(
            "Product_Category",
            "series",
            "Training_End_Date",
            "Forecaster",
            "Drivers_Used_Flag",
            explode(
                arrays_zip(
                    "Driver_STDDEV_Coefficients",
                    "Driver_Impacts"
                )
            ).alias("driver")
        )
        .select(
            "Product_Category",
            "series",
            "Training_End_Date",
            "Forecaster",
            "Drivers_Used_Flag",
            col("driver.Driver_STDDEV_Coefficients._1").alias("Driver"),
            col("driver.Driver_STDDEV_Coefficients._2").alias("STDDEV_Coefficient"),
            col("driver.Driver_Impacts._2").alias("Driver_Impact")
        )
    )

    ## selection based on Topline/Middle for driver
    drv_item = 0 if Topline else 1

    ## filtering for the best version of the driver instead of taking all versions (lags and other)
    w_drv = Window().partitionBy('series','DRV').orderBy(desc(abs('Driver_Impact')))
    temp = (
        drivers_df
        .withColumn('DRV', split(col('Driver'), "__").getItem(drv_item))
        .withColumn('drv_rank', row_number().over(w_drv))
        .filter(col('drv_rank')==1)
    )

    ## filtering to take the top 5 drivers per series
    w_series = Window().partitionBy('series').orderBy(desc(abs("Driver_Impact")))
    arimax_final = (
        temp
        .withColumn('series_drv_rank', row_number().over(w_series))
        .filter(col('series_drv_rank')<=5)
    )
    return arimax_final

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

temp_eval = spark.read.table("Sales_Forecasting.Topline_Eval.driver_analysis")
display(temp_eval.limit(1))
# display(
#     temp_eval
#     .filter(col("Drivers_Used_Flag")=="Automated_Drivers")
#     .groupBy('DRV')
#     .agg(count('*').alias('drv_count'))
#     .orderBy(desc('drv_count'))
# )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(
    temp_eval
    .filter(col('Drivers_Used_Flag')=='Manual_Drivers')
    .groupBy('series','DRV')
    .agg(count('*').alias('drv_count'))
    .orderBy(asc("series"), desc('drv_count'))
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


temp = (
    temp_eval
    .withColumn('Region', split(col('series'),'___').getItem(1))
    .withColumn('Product_Category', split(col('series'),'___').getItem(0))
)
display(
    temp
    .filter(col("Drivers_Used_Flag")=="Manual_Drivers")
    .groupBy('Product_Category','DRV')
    .agg(count('*').alias('drv_count'))
    .orderBy(asc('Product_Category'),desc('drv_count'))
)

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

arimax_auto_final = arimax_drv(automated_drv_dir)
display(arimax_auto_final.limit(3))
arimax_manual_final = arimax_drv(manual_drv_dir)
display(arimax_manual_final.limit(3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### SARIMAX

# CELL ********************

def sarimax_drv(drv_dir):
    sarimax_auto = spark.createDataFrame(pd.read_parquet(drv_dir + 'Sarimax_output.parquet'))
    max_training_date = sarimax_auto.agg(max('Training_End_Date').alias('max_training_date')).collect()[0]['max_training_date']
    sarimax_auto = sarimax_auto.filter(col('Training_End_Date')==max_training_date)

    drivers_df = (
        sarimax_auto
        .select(
            "Product_Category",
            "series",
            "Training_End_Date",
            "Forecaster",
            "Drivers_Used_Flag",
            explode(
                arrays_zip(
                    "Driver_STDDEV_Coefficients",
                    "Driver_Impacts"
                )
            ).alias("driver")
        )
        .select(
            "Product_Category",
            "series",
            "Training_End_Date",
            "Forecaster",
            "Drivers_Used_Flag",
            col("driver.Driver_STDDEV_Coefficients._1").alias("Driver"),
            col("driver.Driver_STDDEV_Coefficients._2").alias("STDDEV_Coefficient"),
            col("driver.Driver_Impacts._2").alias("Driver_Impact")
        )
    )

    ## selection based on Topline/Middle for driver
    drv_item = 0 if Topline else 1

    ## filtering for the best version of the driver instead of taking all versions (lags and other)
    w_drv = Window().partitionBy('series','DRV').orderBy(desc(abs('Driver_Impact')))
    temp = (
        drivers_df
        .withColumn('DRV', split(col('Driver'), "__").getItem(drv_item))
        .withColumn('drv_rank', row_number().over(w_drv))
        .filter(col('drv_rank')==1)
    )

    ## filtering to take the top 5 drivers per series
    w_series = Window().partitionBy('series').orderBy(desc(abs("Driver_Impact")))
    sarimax_final = (
        temp
        .withColumn('series_drv_rank', row_number().over(w_series))
        .filter(col('series_drv_rank')<=5)
    )
    return sarimax_final

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

sarimax_auto_final = sarimax_drv(automated_drv_dir)
display(sarimax_auto_final.limit(1))

sarimax_manual_final = sarimax_drv(manual_drv_dir)
display(sarimax_manual_final.limit(1))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Prophet

# CELL ********************

def prophet_drv(drv_dir):
    prophet_auto = spark.createDataFrame(pd.read_parquet(drv_dir + 'Prophet_output.parquet'))
    max_training_date = prophet_auto.agg(max('Training_End_Date').alias('max_training_date')).collect()[0]['max_training_date']
    prophet_auto = prophet_auto.filter(col('Training_End_Date')==max_training_date)

    drivers_df = (
        prophet_auto
        .select(
            "Product_Category",
            "series",
            "Training_End_Date",
            "Forecaster",
            "Drivers_Used_Flag",
            explode(
                arrays_zip(
                    "Driver_STDDEV_Coefficients",
                    "Driver_Impacts"
                )
            ).alias("driver")
        )
        .select(
            "Product_Category",
            "series",
            "Training_End_Date",
            "Forecaster",
            "Drivers_Used_Flag",
            col("driver.Driver_STDDEV_Coefficients._1").alias("Driver"),
            col("driver.Driver_STDDEV_Coefficients._2").alias("STDDEV_Coefficient"),
            col("driver.Driver_Impacts._2").alias("Driver_Impact")
        )
    )

    ## selection based on Topline/Middle for driver
    drv_item = 0 if Topline else 1

    ## filtering for the best version of the driver instead of taking all versions (lags and other)
    w_drv = Window().partitionBy('series','DRV').orderBy(desc(abs('Driver_Impact')))
    temp = (
        drivers_df
        .withColumn('DRV', split(col('Driver'), "__").getItem(drv_item))
        .withColumn('drv_rank', row_number().over(w_drv))
        .filter(col('drv_rank')==1)
    )

    ## filtering to take the top 5 drivers per series
    w_series = Window().partitionBy('series').orderBy(desc(abs("Driver_Impact")))
    prophet_final = (
        temp
        .withColumn('series_drv_rank', row_number().over(w_series))
        .filter(col('series_drv_rank')<=5)
    )
    return prophet_final

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

prophet_automated_drv = prophet_drv(automated_drv_dir)
display(prophet_automated_drv.limit(1))

prophet_manual_drv = prophet_drv(manual_drv_dir)
display(prophet_manual_drv.limit(1))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### XGBoost

# CELL ********************

def xgb_drv(drv_dir):
    xgboost_auto = spark.createDataFrame(pd.read_parquet(drv_dir + 'XGBoost_SHAP_FORECAST.parquet'))

    for c in xgboost_auto.columns:
        xgboost_auto = xgboost_auto.withColumnRenamed(c, c.replace(".", "_"))

    drv_cols = []
    fixed_cols = ['Product_Category_idx', 'series_idx', 'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
                'lag_18', 'roll_mean_3', 'roll_mean_6', 'roll_mean_12', 'roll_std_3', 'roll_std_6', 'roll_std_12',
                'month', 'quarter', 'year', 'month_sin', 'month_cos', 'Forecast_Horizon', 'Product_Category', 'series',
                'Region', 'Region_idx']
    
    for cols in xgboost_auto.columns:
        if cols not in drv_cols and cols not in fixed_cols:
            drv_cols.append(cols)

    id_cols = ['series', 'Forecast_Horizon']
    if not Topline:
        id_cols.append("Region")

    ## selection based on Topline/Middle for driver
    drv_item = 0 if Topline else 1

    xgb_auto_unpivot = (
        xgboost_auto
        .unpivot(
            ids=id_cols,
            values=drv_cols,
            variableColumnName="Driver",
            valueColumnName="SHAP_Value"
        )
    )
    xgb_auto_unpivot = (
        xgb_auto_unpivot
        .groupBy('series','Driver')
        .agg(avg('SHAP_Value').alias('SHAP_Value'))
        .withColumn("DRV", split(col('Driver'), "__").getItem(drv_item))
    )



    ## filtering for the best version of the driver instead of taking all versions (lags and other)
    w_drv = Window().partitionBy('series','DRV').orderBy(desc(abs('SHAP_Value')))
    temp = (
        xgb_auto_unpivot
        .withColumn('drv_rank', row_number().over(w_drv))
        .filter(col('drv_rank')==1)
    )

    ## filtering to take the top 5 drivers per series
    w_series = Window().partitionBy('series').orderBy(desc(abs("SHAP_Value")))
    xgb_final = (
        temp
        .withColumn('series_drv_rank', row_number().over(w_series))
        .filter(col('series_drv_rank')<=5)
    )

    return xgb_final

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

xgb_auto_drv = xgb_drv(automated_drv_dir).withColumn('Drivers_Used_Flag', lit('Automated_Drivers'))
display(xgb_auto_drv.limit(1))

xgb_manual_drv = xgb_drv(manual_drv_dir).withColumn("Drivers_Used_Flag", lit("Manual_Drivers"))
display(xgb_manual_drv.limit(1))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### TFT

# CELL ********************

def tft_drv(drv_dir):
    tft_auto = spark.createDataFrame(pd.read_parquet(drv_dir + 'TFT_decoder_Output.parquet'))
    max_training_date = tft_auto.agg(max('Training_End_Date').alias('max_training_date')).collect()[0]['max_training_date']
    tft_auto = (
        tft_auto
        .filter(col('Training_End_Date')==max_training_date)
        .groupBy('series', 'feature')
        .agg(
            avg('importance').alias('importance'),
            avg('importance_pct').alias('importance_pct')
        )
        .withColumnRenamed('feature','Driver')
    )


    if Topline:
        tft_auto = tft_auto.withColumn('DRV', col('Driver'))
    else:
        tft_auto = tft_auto.withColumn("DRV", split(col('Driver'), "__").getItem(1))

    ## filtering for the best version of the driver instead of taking all versions (lags and other)
    w_drv = Window().partitionBy('series','DRV').orderBy(desc(abs('importance_pct')))
    temp = (
        tft_auto
        .withColumn('drv_rank', row_number().over(w_drv))
        .filter(col('drv_rank')==1)
    )

    ## filtering to take the top 5 drivers per series
    w_series = Window().partitionBy('series').orderBy(desc(abs("importance_pct")))
    tft_final = (
        temp
        .withColumn('series_drv_rank', row_number().over(w_series))
        .filter(col('series_drv_rank')<=5)
    )
    return tft_final


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

tft_auto_drv = tft_drv(automated_drv_dir).withColumn('Drivers_Used_Flag', lit('Automated_Drivers'))
display(tft_auto_drv.limit(1))

tft_manual_drv = tft_drv(manual_drv_dir).withColumn('Drivers_Used_Flag', lit('Manual_Drivers'))
display(tft_manual_drv.limit(1))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Joining DFs

# CELL ********************

def join_dfs(arimax, sarimax, prophet, xgb, tft, level):
    auto_drv_final =(
        arimax.select('series','Forecaster','Drivers_Used_Flag', 'Driver','DRV')
        .unionByName(
            sarimax_manual_final.select('series','Forecaster','Drivers_Used_Flag', 'Driver','DRV'), allowMissingColumns=True
        )
        .unionByName(
            prophet.select('series','Forecaster','Drivers_Used_Flag', 'Driver','DRV'), allowMissingColumns=True
        )
        .unionByName(
            xgb.withColumn("Forecaster", lit("XGBoost")), allowMissingColumns=True
        )
        .unionByName(
            (
                tft
                .withColumn("Forecaster", lit("TFT"))
            ), 
            allowMissingColumns=True
        )
    ).select('series','Forecaster', 'Drivers_Used_Flag','Driver','DRV').withColumn('Forecast_Level', lit(level))

    return auto_drv_final

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

level = ''
if Topline:
    level="Topline"
else:
    level = "Middle"


auto_final = join_dfs(arimax_auto_final, sarimax_auto_final, prophet_automated_drv, xgb_auto_drv, tft_auto_drv, level)
auto_final = auto_final.filter(~col('DRV').isin('year','quarter','month_cos','month_sin'))
display(auto_final.limit(3))

manual_final = join_dfs(arimax_manual_final, sarimax_manual_final, prophet_manual_drv, xgb_manual_drv, tft_manual_drv, level)
manual_final = manual_final.filter(~col('DRV').isin('year','quarter','month_cos','month_sin'))
display(manual_final.limit(3))


final_df = auto_final.unionByName(manual_final)
display(final_df.limit(3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

final_df.write.mode('overwrite').parquet(output_dir+'Driver_Analysis.parquet')
final_df.write.format('delta').mode('overwrite').saveAsTable(output_table + "driver_analysis")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

temp = spark.read.parquet('abfss://991f5e4b-c174-4ff2-992e-feb17d49d25a@onelake.dfs.fabric.microsoft.com/22746de3-183e-4327-a844-dceda0b7165c/Files/Eval/Middle/Driver_Analysis.parquet')
temp.write.format('delta').mode('overwrite').saveAsTable("Sales_Forecasting.Middle_Eval.driver_analysis")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Looking into the outputs

# CELL ********************

output_table

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

final_df = spark.read.parquet(output_dir+"Driver_Analysis.parquet")
display(final_df.filter(col('Forecaster')=='TFT'))

display(final_df.filter(col('Forecaster')=='XGBoost'))

display(final_df.filter(col('Forecaster')=='Prophet'))

display(final_df.filter(col('Forecaster')=='ARIMAX'))

display(final_df.filter(col('Forecaster')=='SARIMAX'))


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
