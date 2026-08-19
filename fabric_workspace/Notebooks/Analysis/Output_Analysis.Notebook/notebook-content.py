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

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## Parameters
Topline = False

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
    temp_file = spark.read.parquet(file_dir).select(*ACT_GRP_COLS, 'Training_End_Date','Date','Forecast_Horizon','Forecast','Forecaster','Drivers_Used_Flag')
    all_forecast_dfs.append(temp_file)

unioned_dfs = reduce(
    lambda df1, df2: df1.unionByName(df2),
    all_forecast_dfs
)
unioned_dfs.cache()



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

display(eval_table)

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
            avg("Forecast_Error").alias("ME"),
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
