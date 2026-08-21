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
# META     },
# META     "warehouse": {
# META       "known_warehouses": []
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

# PARAMETERS CELL ********************

## Parameters to set from the pipeline
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

actuals_table = config_obj['cutoff_data_table']
initial_target_col = config_obj['target_col']

DRV_GRP_COLS = config_obj['manual_drv_grp_cols']
ACT_GRP_COLS = config_obj['manual_act_grp_cols']

## DRIVER DIRECTORY
## both base_dirs are found within the config file
if driver_status == 'Manual_Drivers':
    selected_driver_dir = config_obj['manual_features_output_dir']
else:
    selected_driver_dir = config_obj['classic_automated_features_output_dir']

parquet_dir = config_obj['forecast_parquet_base_dir']
Seasonal_Baseline_dir = parquet_dir + "Seasonal_Baseline_Output.parquet"



## Model Parameters
STEP_SIZE = Classical_STEP_SIZE
## The following parameters are set within the config file directly
# FORECAST_HORIZON | SEASONAL_PERIODS | MIN_TRAIN | driver_table


## used for renaming for consistency
target_col = 'Value'

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

actuals = spark.read.table(actuals_table).withColumnRenamed(initial_target_col,target_col)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def create_training_end_dates(
    df,
    group_cols,
    min_train,
    forecast_horizon
):
    ## finding global maximum date per series
    series_bounds = (
        df.groupBy(*group_cols)
        .agg(
            min('Date').alias('series_min_date'),
            max('Date').alias('series_max_date')
        )
    )

    ## latest possible forecast origin using forecast_horizon as constraint

    series_bounds = (
        series_bounds
        .withColumn(
            'series_latest_start_date',
            add_months(col('series_max_date'),-forecast_horizon)
            )
    )

    ## earliest training cutoff due to minimum training history
    series_bounds = (
        series_bounds
        .withColumn(
            'series_earliest_start_date',
            add_months(col('series_min_date'),min_train)
        )
    )


    training_dates = (
        series_bounds
        .select(*group_cols, explode(
            sequence(
                col('series_earliest_start_date'),
                col('series_latest_start_date'),
                expr("INTERVAL 1 MONTH")
            )
        ).alias('training_end_date')
        )
    )

    return training_dates



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql import functions as F
from pyspark.sql.window import Window


def seasonal_naive_forecast(
    df,
    training_end_dates,
    group_cols,
    target_col,
    forecast_horizon
):

    ############################################################
    # Create future forecast dates
    ############################################################

    forecast_dates = (
        training_end_dates
        .select(*group_cols, "training_end_date")
        .withColumn(
            "Date",
            F.explode(
                F.sequence(
                    F.add_months(F.col("training_end_date"), 1),
                    F.add_months(F.col("training_end_date"), forecast_horizon),
                    F.expr("INTERVAL 1 MONTH")
                )
            )
        )
        .withColumn(
            "forecast_horizon",
            F.months_between(
                F.col("Date"),
                F.col("training_end_date")
            ).cast("int")
        )
    )


    ############################################################
    # Historical data before each training cutoff
    ############################################################


    history = (
        df.alias("actuals")
        .join(
            training_end_dates
            .select(*group_cols, "training_end_date")
            .alias("cuts"),
            on=group_cols,
            how="inner"
        )
        .filter(
            F.col("actuals.Date") <= F.col("cuts.training_end_date")
        )
        .select(
            *[
                F.col(f"actuals.{c}").alias(c)
                for c in group_cols
            ],
            F.col("cuts.training_end_date"),
            F.col("actuals.Date"),
            F.col(f"actuals.{target_col}")
        )
    )


    ############################################################
    # Create seasonal lag
    #
    # Seasonal naive:
    # forecast(t+h) = actual(t+h-12)
    ############################################################

    seasonal_window = (
        Window
        .partitionBy(
            *group_cols,
            "training_end_date"
        )
        .orderBy("Date")
    )


    history = (
        history
        .withColumn(
            "lag_12",
            F.lag(
                F.col(target_col),
                12
            )
            .over(seasonal_window)
        )
    )


    ############################################################
    # Lookup table containing historical seasonal values
    ############################################################

    seasonal_lookup = (
        history
        .select(
            *group_cols,
            "training_end_date",
            F.col("Date").alias("seasonal_reference_date"),
            F.col("lag_12")
        )
    )


    ############################################################
    # Forecast date requires value 12 months prior
    ############################################################

    forecast_dates = (
        forecast_dates
        .withColumn(
            "seasonal_reference_date",
            F.add_months(
                F.col("Date"),
                -12
            )
        )
    )


    ############################################################
    # Join forecast periods to historical seasonal values
    ############################################################

    join_condition = [
        forecast_dates[c] == seasonal_lookup[c]
        for c in group_cols
    ]

    join_condition += [
        forecast_dates["training_end_date"] 
        == seasonal_lookup["training_end_date"],

        forecast_dates["seasonal_reference_date"]
        == seasonal_lookup["seasonal_reference_date"]
    ]

    forecasts = (
        forecast_dates.alias("fcst")
        .join(
            seasonal_lookup.alias("seasonal"),
            on=join_condition,
            how="left"
        )
        .select(
            *[
                F.col(f"fcst.{c}")
                for c in group_cols
            ],
            F.col("fcst.training_end_date"),
            F.col("fcst.Date"),
            F.col("fcst.forecast_horizon"),
            F.col("seasonal.lag_12").alias("forecast")
        )
    )


    ############################################################
    # Return final benchmark forecast
    ############################################################

    forecasts = (
        forecasts
        .orderBy(
            *group_cols,
            "training_end_date",
            "Date"
        )
    )


    return forecasts


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

training_dates = create_training_end_dates(actuals,ACT_GRP_COLS, MIN_TRAIN, FORECAST_HORIZON)

forecasts = seasonal_naive_forecast(
    actuals,
    training_dates,
    ACT_GRP_COLS,
    target_col,
    FORECAST_HORIZON
)

t_window = Window.partitionBy(*ACT_GRP_COLS, 'training_end_date').orderBy('Date')
forecasts = forecasts.withColumn('forecast_v2', lag(col('forecast'),12).over(t_window))
forecasts = forecasts.withColumn('forecast_final', coalesce(col('forecast'),col('forecast_v2')))
forecasts = forecasts.drop('forecast','forecast_v2').withColumnRenamed('forecast_final','forecast')

forecasts = (
    forecasts
    .withColumn("Forecaster", lit("Seasonal_Baseline"))
    .withColumn("Drivers_Used_Flag", lit("No_Drivers"))
    .withColumnsRenamed({
        "training_end_date": "Training_End_Date",
        "forecast_horizon": "Forecast_Horizon",
        'forecast': "Forecast"
    })
)

forecasts.write.mode('overwrite').parquet(Seasonal_Baseline_dir)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
