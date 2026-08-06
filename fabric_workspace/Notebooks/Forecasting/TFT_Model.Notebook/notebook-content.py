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
# META     "environment": {
# META       "environmentId": "2448fb99-ede1-b457-4606-3f14733f8d02",
# META       "workspaceId": "00000000-0000-0000-0000-000000000000"
# META     }
# META   }
# META }

# CELL ********************

# Welcome to your new notebook
import numpy as np
from dateutil.relativedelta import relativedelta
import math
from pyspark.sql import DataFrame, functions as F, Window
from pyspark.ml.feature import StringIndexer, VectorAssembler
import xgboost as xgb
import pandas as pd
from pyspark.sql.types import *
from pyspark.sql.functions import *
import operator
import builtins
from functools import reduce
from sklearn.metrics import mean_absolute_error
import optuna
import re

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting import TimeSeriesDataSet

from optuna.integration import PyTorchLightningPruningCallback


import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting.data import GroupNormalizer


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# PARAMETERS CELL ********************

## default parameters / parameters to pass in from the pipeline
run_tft = True
Topline = False
rerun_historical_forecasts = False
driver_status = "Automated_Drivers"    ## options: "No_Drivers", "Manual_Drivers", "Automated_Drivers" 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print(f"run_tft: {run_tft}")
print(f"Topline: {Topline}")
print(f"rerun_historical_forecasts: {rerun_historical_forecasts}")
print(f"driver_status: {driver_status}")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

VALID_DRIVER_STATUS = ["No_Drivers", "Manual_Drivers", "Automated_Drivers"]
if driver_status not in VALID_DRIVER_STATUS:
    raise ValueError(
        f"Invalid driver_status={driver_status}. "
        f"Expected one of {VALID_DRIVER_STATUS}"
    )



# ==========================================================
# CONFIG
# ==========================================================
manual_features = "/lakehouse/default/Files/Driver_Analysis/Final_Feature_Selection/"
automated_features = "/lakehouse/default/Files/Automated_Driver_Analysis/"
parquet_dir = "abfss://991f5e4b-c174-4ff2-992e-feb17d49d25a@onelake.dfs.fabric.microsoft.com/22746de3-183e-4327-a844-dceda0b7165c/Files/Forecasting"

if Topline:
    actuals_table = "Sales_Forecasting.silver.topline_cutoff_data"
    initial_target_col = "Quantity"
    DRV_GRP_COLS = ['Indicator']
    ACT_GRP_COLS = ['Product_Category','series']
    if driver_status == 'Manual_Drivers':
        selected_driver_dir = manual_features + "final_features_topline.csv"
    else:
        selected_driver_dir = automated_features + "Topline/topline_xgboost_selected_feature.xlsx"
    parquet_dir = parquet_dir + "/Topline/" + driver_status + "/"
    TFT_forcast_dir = parquet_dir + "TFT_Forecast_Output.parquet"
    TFT_static_dir = parquet_dir + "TFT_static_Output.parquet"
    TFT_decoder_dir = parquet_dir + "TFT_decoder_Output.parquet"
    TFT_attention_dir = parquet_dir + "TFT_attention_Output.parquet"
else:
    actuals_table = "Sales_Forecasting.silver.middle_cutoff_data"
    initial_target_col = 'Quantity'
    DRV_GRP_COLS = ['Region','Indicator']
    ACT_GRP_COLS = ['Product_Category', 'Region','series']
    if driver_status == 'Manual_Drivers':
        selected_driver_dir = manual_features + "final_features_middle.csv"
    else:
        selected_driver_dir = automated_features + "Middle/middle_xgboost_selected_features.xlsx"
    parquet_dir = parquet_dir + "/Middle/" + driver_status + "/"
    TFT_forcast_dir = parquet_dir + "TFT_Forecast_Output.parquet"
    TFT_static_dir = parquet_dir + "TFT_static_Output.parquet"
    TFT_decoder_dir = parquet_dir + "TFT_decoder_Output.parquet"
    TFT_attention_dir = parquet_dir + "TFT_attention_Output.parquet"

FORECAST_HORIZON = 18
SEASONAL_PERIODS = 12
MIN_TRAIN = 36
#STEP_SIZE = 3  # is dynamically set to only do 6 historical forecasting run to avoid long runtimes

               
TUNE_HOLDOUT_MONTHS = FORECAST_HORIZON
N_OPTUNA_TRIALS =  50
OPTUNA_SEED = 42
ENCODER_LENGTH = 36


driver_table = "Sales_Forecasting.silver.compiled_drivers"
target_col = 'Value'
if driver_status == "No_Drivers":
    drivers_used = False
else:
    drivers_used = True

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

### Reading Data
actuals = (
    spark.read.table(actuals_table)
    .withColumnRenamed(initial_target_col, target_col)
    .fillna(0.0,subset='Value')
    .withColumn(target_col, greatest(col(target_col),lit(0)))
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def add_future_months(df, ACT_GRP_COLS, date_col='Date', horizon=18):
    """
        add future month records for each unique series
    """
    df = df.withColumn(date_col, to_date(col(date_col)))

    global_max_date = df.agg(F.max(date_col).alias('last_date')).collect()[0]['last_date']
    print(f"using global max date across all series as the forecast origin: {global_max_date}")

    last_dates = (
        df.select(*ACT_GRP_COLS).distinct()
        .withColumn('last_date', lit(global_max_date))
    )
    future = (
        last_dates
        .withColumn('offset', explode(sequence(lit(1), lit(horizon))))
        .withColumn(date_col, add_months(col('last_date'), col('offset')))
        .drop('last_date','offset')
    )
    missing_cols = [c for c in df.columns if c not in ACT_GRP_COLS + [date_col]]
    for c in missing_cols:
        future = future.withColumn(c, lit(None).cast(df.schema[c].dataType))
    future = future.select(df.columns)
    return df.unionByName(future)




actuals_fh_populated = add_future_months(actuals, ACT_GRP_COLS)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

if drivers_used:
    compiled_drivers = spark.read.table(driver_table)
    aggregated_drivers = compiled_drivers.groupBy(*DRV_GRP_COLS,'Date').agg(sum('Value').alias('Value'))
    if 'Indicator' in DRV_GRP_COLS and len(DRV_GRP_COLS) > 1:
        world_agg = compiled_drivers.groupBy('Indicator','Date').agg(sum("Value").alias("Value"))
        world_agg = world_agg.withColumn("Indicator", concat_ws("__", col("Indicator"), lit("WORLD")))
        for cols in DRV_GRP_COLS:
            if cols != 'Indicator':
                distinct_vals = compiled_drivers.select(cols).distinct()
                world_agg = world_agg.crossJoin(distinct_vals)
                print(f"{cols} to the world agg using cross join of distinct values from the drivers data")
        aggregated_drivers = aggregated_drivers.unionByName(world_agg)
    else:
        print('world agg already done')
    if 'csv' in selected_driver_dir:
        print('reading csv')
        selected_drivers = spark.createDataFrame(
            pd.read_csv(selected_driver_dir).drop(columns="Unnamed: 0", errors="ignore")
        )
    elif 'xlsx' in selected_driver_dir:
        print('reading excel')
        selected_drivers = spark.createDataFrame(
            pd.read_excel(selected_driver_dir).drop(columns="Unnamed: 0", errors="ignore")
        )
        if Topline:
            selected_drivers = (
                selected_drivers
                .withColumn('Lag', split(col('Feature'),"__").getItem(1))
                .withColumn('Indicator', split(col('Feature'),"__").getItem(0))
                .drop('Feature','Lag')      ## Dropping LAG as TFT will take unlagged values
            )
        else:
            selected_drivers = (
                selected_drivers
                .withColumn('Lag', split(col("Feature"),"__").getItem(2))
                .withColumn("Indicator", split(col("Feature"), "__").getItem(1))
                .drop('Feature','Lag')      ## Dropping LAG as TFT will take unlagged values
            )
    else:
        raise ValueError(f"Unsupported file type. expected csv or xlsx but received: {selected_driver_dir}")


    check_cols = list(dict.fromkeys(ACT_GRP_COLS+DRV_GRP_COLS))
    check_cols.remove('series')

    print(f"selected_driver num distinct records: {selected_drivers.select(*check_cols).distinct().count()}")
    print(f"agg_drivers records: {aggregated_drivers.select(*DRV_GRP_COLS).distinct().count()}")

    joined_driver_data = (
        broadcast(selected_drivers).join(aggregated_drivers, [*DRV_GRP_COLS], 'left')
    )
    print(f"joined records: {joined_driver_data.select(*check_cols).distinct().count()}")

    if (selected_drivers.select(*check_cols).distinct().count()) != (joined_driver_data.select(*check_cols).distinct().count()):
        raise ValueError("There is a mismatch and the number of combinations has changed post selected driver & aggregated driver join")

    joined_driver_data = (
        joined_driver_data
        .withColumnsRenamed({"Value":"driver_value","Date":"driver_date"})
    )

    join_cols = ACT_GRP_COLS.copy()
    join_cols.remove('series')
    conditions = [col(f"a.{c}") == col(f"d.{c}") for c in join_cols]
    conditions.append(col("a.Date") == col("d.driver_date"))
    join_cond = reduce(operator.and_, conditions)
    actuals_w_drivers = (
        actuals_fh_populated.alias("a")
        .join(joined_driver_data.alias("d"), join_cond, "left")
        .select("a.*", *[col(f"d.{c}") for c in DRV_GRP_COLS if c not in ACT_GRP_COLS], "d.driver_value")
    )
    drop_cols = [c for c in DRV_GRP_COLS if c not in ACT_GRP_COLS]
    actuals_w_drivers = (
        actuals_w_drivers
        .withColumn('feature_col', concat_ws("__", *DRV_GRP_COLS))
        .drop(*drop_cols)
    )
    actuals_fh_populated = actuals_w_drivers
    print('actuals_fh_populated dataframe is now overwritten with a dataframe containing driver data in long format')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## Data Transformations / Processing
def pivot_long_to_wide(sdf):
    if drivers_used:
        wide = (
            sdf.groupBy(*ACT_GRP_COLS,'Date',target_col)
            .pivot('feature_col')
            .agg(first('driver_value'))
        )
        return wide
    else:
        return sdf



# ==========================================================
# STEP 1 — calendar features
# ==========================================================
def add_calendar_features(sdf: DataFrame, date_col: str = "Date") -> DataFrame:
    sdf = sdf.withColumn("_dt", to_date(col(date_col)))
    sdf = sdf.withColumn("month", month("_dt"))
    sdf = sdf.withColumn("quarter", quarter("_dt"))
    sdf = sdf.withColumn("year", year("_dt"))
    sdf = sdf.withColumn("month_sin", sin(col("month") * (2 * np.pi / 12)))
    sdf = sdf.withColumn("month_cos", cos(col("month") * (2 * np.pi / 12)))
    calendar_cols = ['_dt', 'month', 'quarter', 'year', 'month_sin', 'month_cos']
    return sdf, calendar_cols 

    



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

###############################################
# Default hyperparameters
###############################################

def default_tft_params():

    return {
        "hidden_size": 16,
        "attention_head_size": 2,
        "dropout": 0.15,
        "hidden_continuous_size": 8,
        "learning_rate": 1e-3,
        "batch_size":128,
        "gradient_clip_val":0.1,
    }

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

############################################################
# Objective
############################################################
def objective(trial, train_dataset, validation_dataset):
    pl.seed_everything(OPTUNA_SEED)
    print("Another trail/objective has been run")

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-3,
        patience=10,
        mode="min",
    )

    pruning_callback = PyTorchLightningPruningCallback(
        trial,
        monitor="val_loss"
    )

    params = {
        "hidden_size":
            trial.suggest_categorical(
                "hidden_size",      ## controls the size of the neural network representations 
                                    ## higher = more complex relationships and more interactions between variables are learned
                                    ## however higher also leads to increased training time, memory usage and increased risk of overfitting

                ### changing these parameters to reduce the search space of optuna
                # 16,
                # 128,
                # step=16,
                [8,16,32]
            ),
        "attention_head_size":
            trial.suggest_categorical(
                "attention_head_size",      ## multiple heads enable multi-head attention (which previous time steps should the model pay attention to)
                                            ## head 1 (recent months), head 2 (seasonality), head 3 (long term trend)
                                            ## the higher number value for attention head size enables the learning of more patterns
                                            ## However this also leads to more computation, parameters and harder optimization
                [1,2,4]
                ### changing these parameters to reduce the search space and computation of optuna
                # 1,
                # 8,
            ),
        "dropout":
            trial.suggest_float(
                "dropout",
                0.05, 
                0.25,
                ## changing to reduce optuna time
                # 0.05,
                # 0.30,
            ),
        "hidden_continuous_size":
            trial.suggest_categorical(
                "hidden_continuous_size",
                [6,16]
                                ### changing these parameters to reduce the search space and computation of optuna
                # 8,
                # 64,
                # step=8,
            ),
        "learning_rate":
            trial.suggest_float(
                "learning_rate",
                5e-4,
                3e-3,

                                ### changing these parameters to reduce the search space and computation of optuna
                # 1e-4,
                # 1e-2,
                log=True,
            ),
        "gradient_clip_val":
            trial.suggest_categorical(
                "gradient_clip_val",
                [0.01,0.1,0.3],

                                ### changing these parameters to reduce the search space and computation of optuna
                # 0.01,
                # 1.0,
                # log=True,
            ),
        "batch_size":
            trial.suggest_categorical(
                "batch_size",
                [64,128],
                                ### changing these parameters to reduce the search space and computation of optuna
                # [64,128,256],
            ),
    }

    ########################################################
    # DataLoaders
    ########################################################
    print("creating train and val data loaders")
    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(validation_dataset))


    train_loader = train_dataset.to_dataloader(
        train=True,
        batch_size=params["batch_size"],
        num_workers=0,
    )

    val_loader = validation_dataset.to_dataloader(
        train=False,
        batch_size=params["batch_size"],
        num_workers=0,
    )

    ########################################################
    # Build model
    ## this is building the model or the architecture of the neural network
    ## this builds out the layers, weights, and input feature ingestion / how to produce predictions
    ## however it has not trained on the data yet
    ########################################################
    print("building TFT model based upon train dataset and trial params")
    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=params["learning_rate"],
        hidden_size=params["hidden_size"],
        attention_head_size=params["attention_head_size"],
        hidden_continuous_size=params["hidden_continuous_size"],
        dropout=params["dropout"],
        loss=QuantileLoss(),
        log_interval=10,
    )

    ########################################################
    # Trainer
    ## the trainer does not learn
    ## the trainer is the manager of the process and outlines the following
    ## the data, how many times to run, how to calc errors, update, and validate performance
    ########################################################
    print("creating the trainer")
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs= 50, #15,
        limit_train_batches=1.0,
        gradient_clip_val=params["gradient_clip_val"],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[
            early_stop_callback,
            pruning_callback
            ]
    )

    ########################################################
    # Train
    ## this is where the training/learning occurs
    ## the trainer receives a batch
    ## the dataloader provides the examples in line with batch size
    ## the model predicts based on this input and calcualtes loss
    ## the model then backpropagates (what weight caused this error) and then updates/adjusts
    ########################################################
    print("fitting the model")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    ########################################################
    # Evaluate
    ## this is just extracting the val_loss metric from the trainer
    ## how well did this trained model perform on unseen data?
    ########################################################
    print("Beginning model evaluation")
    # score = trainer.validate(
    #     model,
    #     val_loader,
    #     verbose=False,
    # )[0]["val_loss"]

    # print('returning the score of the trail/run')
    # print(f'score was {score}')

    # score = trainer.callback_metrics["val_loss"].item()

    if "val_loss" not in trainer.callback_metrics:
        raise RuntimeError(
            f"Validation loss missing. Metrics available: {trainer.callback_metrics}"
        )

    score = trainer.callback_metrics["val_loss"].item()

    print("Returning validation loss")
    print(f"score was {score}")

    print('END OF TRIAL')
    print("="*80)
    return score

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def tune_tft_hyperparameters(
    training_df,
    encoder_length,
    prediction_length,
    static_categoricals,
    time_varying_known_reals,
    time_varying_unknown_reals,
    target,
    group_ids,
    n_trials=30,
    seed=42,
):
    """
    Tunes TFT hyperparameters ONE TIME.
    Uses the final forecast horizon as a chronological validation set.
    Returns
    -------
    best_params : dict
    """
    print('creating chronological split')
    ############################################################
    # Create chronological split
    ############################################################
    ## cutoff date is set by max date - the prediction length (Forecast Horizon)
    max_idx = training_df.time_idx.max()
    cutoff = max_idx - prediction_length

    train_df = training_df[
        training_df.time_idx <= cutoff
    ]


    validation_df = training_df[
        training_df.time_idx > cutoff - encoder_length
    ]



    print(f'split with cutoff date of: {cutoff}')


    print('building train timeseriesdataset')
    ############################################################
    # Build TimeSeriesDataSet
    ############################################################
    train_dataset = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=target,
        group_ids=group_ids,
        max_encoder_length=encoder_length,
        max_prediction_length=prediction_length,
        static_categoricals=static_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(
            groups=group_ids,
            transformation="softplus",
        ),
        allow_missing_timesteps=True,
    )

    print("building validation timeseriesdataset")
    validation_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        validation_df,
        predict=True,
        stop_randomization=True,
    )


    ############################################################
    # Run Optuna
    ############################################################

    print('running optuna')
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
        ),
    )
    print('beginning the optuna study')
    
    study.optimize(
        lambda trial: objective(trial, train_dataset, validation_dataset),
        n_trials=n_trials,
    )

    print("Best Validation Loss:", study.best_value)

    print(study.best_params)

    return study.best_params

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def run_tft_forecasts(best_params, hist_df, forecast_df, time_varying_known_reals):

    ## Creating the training dataset
    train_dataset = TimeSeriesDataSet(
        hist_df,
        time_idx='time_idx',
        target=target_col,
        group_ids=ACT_GRP_COLS,
        max_encoder_length=ENCODER_LENGTH,
        max_prediction_length=FORECAST_HORIZON,
        static_categoricals=ACT_GRP_COLS,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=[target_col],
        target_normalizer=GroupNormalizer(
            groups=ACT_GRP_COLS,
            transformation="softplus",
        ),
        allow_missing_timesteps=True,
    )

    train_loader = train_dataset.to_dataloader(
        train=True,
        batch_size = best_params['batch_size']
    )

    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate = best_params["learning_rate"],
        hidden_size=best_params["hidden_size"],
        attention_head_size=best_params["attention_head_size"],
        hidden_continuous_size=best_params["hidden_continuous_size"],
        dropout=best_params["dropout"],
        loss=QuantileLoss(),
    )

    trainer = pl.Trainer(
        max_epochs=50,
        gradient_clip_val=best_params["gradient_clip_val"],
        enable_checkpointing=True,
        enable_progress_bar=False        
    )

    trainer.fit(
        model,
        train_loader,
    )


    ## Creating Future DataSet / Predictions
    future_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        forecast_df,
        predict=True,
        stop_randomization=True,
    )

    future_loader = future_dataset.to_dataloader(
        train=False,
        batch_size=best_params['batch_size'],
        num_workers=0
    )

    raw_predictions= model.predict(
        future_loader,
        mode="raw",
        return_x=True,
        return_index=True
    )



    ## point forecasts
    point_predictions = model.to_prediction(raw_predictions.output)
    point_predictions = point_predictions.detach().cpu().numpy()


    # Index describing each prediction
    prediction_index = raw_predictions.index.reset_index(drop=True)

    # -------------------------------
    # Flatten point forecasts
    # -------------------------------

    forecasted_rows = []

    for i, row in prediction_index.iterrows():
        for h in range(point_predictions.shape[1]):
            forecasted_rows.append({
                **row.to_dict(),
                "forecast_horizon": h + 1,
                "prediction": point_predictions[i, h]
            })

    forecasted_df = spark.createDataFrame(pd.DataFrame(forecasted_rows))

    org_forecast_df = spark.createDataFrame(forecast_df)

    joined_forecasted_df = (
        forecasted_df.withColumn('time_idx', col('time_idx')-1)
        .join(
            broadcast(org_forecast_df.select('Date','time_idx').distinct()),
            'time_idx',
            'inner'
        )
    )

    joined_forecasted_df = (
        joined_forecasted_df
        .withColumn('End_Training_Date', col('Date'))
        .withColumn('Date', add_months(col('Date'),col('forecast_horizon')))
        .withColumn("Forecaster", lit("TFT_Global"))
        .withColumn("Drivers_Used_Flag", lit(driver_status))
        .drop('time_idx')
    )



    prediction_index = (
        prediction_index
        .assign(time_idx=lambda x: x["time_idx"] - 1)
        .merge(
            forecast_df[
                ACT_GRP_COLS + ["time_idx", "Date"]
            ].drop_duplicates(),
            on=ACT_GRP_COLS + ["time_idx"],
            how="left"
        )
        .rename(
            columns={
                "Date": "End_Training_Date"
            }
        )
    )

    prediction_index['time_idx'] = prediction_index['time_idx']+1



    # ------------------------------------------------------------
    # STATIC VARIABLE IMPORTANCE
    # Shape:
    # [batch_size, number_static_variables]
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # STATIC VARIABLE IMPORTANCE
    # ------------------------------------------------------------

    static_values = (
        raw_predictions.output["static_variables"]
        .detach()
        .cpu()
        .numpy()
    )


    # Remove singleton dimension only
    if static_values.ndim == 3 and static_values.shape[1] == 1:
        static_values = static_values.squeeze(axis=1)

    if static_values.ndim == 1:
        static_values = static_values.reshape(1, -1)

    static_variables = future_dataset.static_categoricals


    static_rows = []

    for i, row in prediction_index.iterrows():

        for v, variable in enumerate(static_variables):

            static_rows.append(
                {
                    **row.to_dict(),
                    "variable": variable,
                    "importance": float(static_values[i,v])
                }
            )


    static_importance_df = spark.createDataFrame(
        pd.DataFrame(static_rows)
    )

    static_importance_df = static_importance_df.drop('time_idx')





    # ------------------------------------------------------------
    # DECODER VARIABLE IMPORTANCE
    # ------------------------------------------------------------

    decoder_values = (
        raw_predictions.output["decoder_variables"]
        .detach()
        .cpu()
        .numpy()
    )



    # remove target dimension
    if decoder_values.ndim == 4 and decoder_values.shape[2] == 1:
        decoder_values = decoder_values.squeeze(axis=2)


    decoder_variables = future_dataset.time_varying_known_reals




    decoder_rows = []

    assert decoder_values.shape[-1] == len(decoder_variables)
    for i, row in prediction_index.iterrows():

        for h in range(decoder_values.shape[1]):

            for v, variable in enumerate(decoder_variables):

                decoder_rows.append(
                    {
                        **row.to_dict(),
                        "forecast_horizon": h + 1,
                        "variable": variable,
                        "importance": float(
                            decoder_values[i,h,v]
                        )
                    }
                )


    decoder_importance_df = pd.DataFrame(decoder_rows)


    decoder_importance_df["importance_pct"] = (
        decoder_importance_df
        .groupby(
            [
                *ACT_GRP_COLS,
                "End_Training_Date",
                "forecast_horizon"
            ]
        )["importance"]
        .transform(
            lambda x:
            x / x.sum()
            if x.sum() != 0
            else 0
        )
    )


    decoder_importance_df = spark.createDataFrame(
        decoder_importance_df
    )
    decoder_importance_df = decoder_importance_df.drop('time_idx')


    # ------------------------------------------------------------
    # ATTENTION WEIGHTS
    # Shows which historical periods drove each forecast horizon
    # ------------------------------------------------------------

    attention_values = (
        raw_predictions.output["decoder_attention"]
        .detach()
        .cpu()
        .numpy()
    )

    print("raw attention shape:", attention_values.shape)


    # Remove attention head dimension
    # (batch, heads, horizon, encoder_length)
    # ->
    # (batch, horizon, encoder_length)

    if attention_values.ndim == 4:
        attention_values = attention_values.mean(axis=2)


    print("processed attention shape:", attention_values.shape)

    attention_rows = []

    for i, row in prediction_index.iterrows():

        for h in range(attention_values.shape[1]):

            for lag in range(attention_values.shape[2]):

                attention_rows.append(
                    {
                        **row.to_dict(),
                        "forecast_horizon": h + 1,
                        "encoder_lag": lag + 1,
                        "attention_weight": float(
                            attention_values[i,h,lag]
                        )
                    }
                )


    attention_df = spark.createDataFrame(
        pd.DataFrame(attention_rows)
    )
    attention_df = attention_df.drop('time_idx')



    return (
        joined_forecasted_df,
        static_importance_df,
        decoder_importance_df,
        attention_df
    )



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


## Main Orchestration

def fit_TFT_global(sdf):
    """
    Fits a global TFT model

    rerun_historical_forecasts = True:
        TRUE walk-forward backtesting — for each historical origin date
        (stepped by STEP_SIZE), a model is trained ONLY on data known
        up to that origin
        Origin dates are aligned across ALL series (see "origin date
        selection" block) so a single global model at a given origin
        has every series already past MIN_TRAIN.

    Regardless of that flag, this ALWAYS produces the forward-looking
    forecast for every row where target_col is null.
    """


    sdf = pivot_long_to_wide(sdf)
    ### REPLACING "." WITH "____" IN ORDER TO HAVE THE COLUMN NAMES WORK WITH TFT
    sdf = sdf.toDF(*[c.replace(".", "____") for c in sdf.columns])


    excluded_cols = ACT_GRP_COLS + ['Date',target_col]
    driver_cols = [c for c in sdf.columns if c not in excluded_cols]

    sdf, calendar_cols = add_calendar_features(sdf)

    ## creating time index column based on the Date column by *ACT_GRP_COLS
    min_date = sdf.agg(min("Date")).first()[0]
    sdf = sdf.withColumn(
        'time_idx', 
        months_between(col('Date'), lit(min_date)).cast('int'))

    ## setting final training dataframes for initial hyperparameter tuning & final forecasting
    training_df = sdf.filter(col(target_col).isNotNull())
    training_df = training_df.orderBy(*ACT_GRP_COLS,'time_idx').toPandas()
    training_df[driver_cols] = training_df[driver_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    time_varying_known_reals = driver_cols + calendar_cols
    time_varying_known_reals.remove("_dt")

    ## need to also fill the target_col with 0 prior to ingestion for predictions as NaN/Nulls are not allowed
    full_df = sdf.fillna(0.0, subset=driver_cols)
    full_df = full_df.fillna(0.0, subset=target_col)
    full_df = full_df.orderBy(*ACT_GRP_COLS,'time_idx').toPandas()
    full_df[driver_cols] = full_df[driver_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)



    print("BEGINNING TFT HYPERPARAMETER TUNING")
    print("="*80)


    best_params = tune_tft_hyperparameters(
        training_df,
        encoder_length=ENCODER_LENGTH,
        prediction_length=FORECAST_HORIZON,
        ## Column CLassification
        static_categoricals=ACT_GRP_COLS,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=[target_col],
        target=target_col,
        group_ids=ACT_GRP_COLS,
        n_trials=N_OPTUNA_TRIALS,
        seed=OPTUNA_SEED
    )


    ### leverage run_tft_forecasts for downstream code

    all_forecasts = []
    all_static_importance = []
    all_decoder_importance = []
    all_attention = []





    ## HISTORICAL FORECASTING

    sdf.cache()

    if rerun_historical_forecasts:
        first_date = (
            sdf
            .filter(col(target_col).isNotNull())
            .groupBy(ACT_GRP_COLS)
            .agg(
                min('Date').alias('min_date')
            )
            .withColumn(
                'date_options',
                add_months(col('min_date'), ENCODER_LENGTH)
            )
            .agg(
                max(col('date_options')).alias('first_date')
                ).collect()[0]['first_date']
        )
        max_elig_date = (
            sdf
            .filter(col(target_col).isNotNull())
            .agg(
                max('Date').alias('max_date')
            )
            .withColumn('max_date',add_months(col('max_date'),-1)) ## to ensure that the most recent date is not used for historical forecasts
            .collect()[0]['max_date']
        )

        eligible_dates = (
            sdf
            .filter(
                (col('Date')>=first_date) &
                (col('Date')<=max_elig_date)
            )
            .select('Date').distinct()
        )

        print(first_date)
        print(max_elig_date)
        
        ## ensures that a max of 6 historical forecasts are done
        step_size = math.ceil(eligible_dates.count()/6)

        print(f"the step size as been set to {step_size}")

        ## generating origin dates in range using the step size
        origin_dates = [
            r['Date'] for r in 
            eligible_dates.orderBy('Date').collect()
        ][::step_size]

        if ((origin_dates[1].year - origin_dates[0].year) * 12 + (origin_dates[1].month - origin_dates[0].month))>18:
            raise ValueError("The step size is larger than 18, due to this a historical rerun will not cover all months with forecast values")
        else:
            print('the step size is small enough to ensure coverage')


        ## iterating through the origin dates for historical forecasting
        for origin_date in origin_dates:
            
            hist_df = sdf.filter(col('Date')<=origin_date)

            forecast_end = origin_date + relativedelta(months=18)

            forecast_df = (
                sdf
                .filter(
                    (col('Date')<=forecast_end)
                )
                .withColumn(target_col, lit(0.0).cast('double'))
            )


            hist_df = hist_df.orderBy(*ACT_GRP_COLS,'time_idx').toPandas()
            forecast_df = forecast_df.orderBy(*ACT_GRP_COLS,'time_idx').toPandas()
            hist_df[driver_cols] = hist_df[driver_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            forecast_df[driver_cols] = forecast_df[driver_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

            
            joined_forecasted_df, static_importance_df, decoder_importance_df, attention_df = run_tft_forecasts(best_params, hist_df, forecast_df, time_varying_known_reals)


            all_forecasts.append(joined_forecasted_df)

            all_static_importance.append(static_importance_df)

            all_decoder_importance.append(decoder_importance_df)

            all_attention.append(attention_df)






    ## FUTURE FORECASTING


    print("BEGINNING FINAL FORECASTING ITERATION FOR FUTURE FORECAST HORIZONS")
    print("="*80)
    joined_forecasted_df, static_importance_df, decoder_importance_df, attention_df = run_tft_forecasts(best_params, training_df, full_df, time_varying_known_reals)


    all_forecasts.append(joined_forecasted_df)

    all_static_importance.append(static_importance_df)

    all_decoder_importance.append(decoder_importance_df)

    all_attention.append(attention_df)



    ## CREATION OF FINAL OUTPUT DATASETS


    from functools import reduce


    final_forecasts = reduce(
        lambda df1, df2: df1.unionByName(df2),
        all_forecasts
    )


    final_static_importance = reduce(
        lambda df1, df2: df1.unionByName(df2),
        all_static_importance
    )


    final_decoder_importance = reduce(
        lambda df1, df2: df1.unionByName(df2),
        all_decoder_importance
    )


    final_attention = reduce(
        lambda df1, df2: df1.unionByName(df2),
        all_attention
    )



    return (
        final_forecasts,
        final_static_importance,
        final_decoder_importance,
        final_attention
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

if run_tft:
    print(f"Topline status: {Topline}")
    print(driver_status)
    print(f"run_historical_forecasts: {rerun_historical_forecasts}")

    forecasts, static_importance, decoder_importance, attention = fit_TFT_global(actuals_fh_populated)
    
    forecasts.cache()
    static_importance.cache()
    decoder_importance.cache()
    attention.cache()

    forecasts = forecasts.withColumnRenamed('prediction','Forecast')
    static_importance = static_importance.withColumnRenamed('variable','feature')
    decoder_importance = decoder_importance.withColumnRenamed('variable','feature')


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

if run_tft:
    if rerun_historical_forecasts:
        print(f"overwriting parquet with new historical forecast values: {TFT_forcast_dir}")
        forecasts.write.mode('overwrite').parquet(TFT_forcast_dir)
        static_importance.write.mode('overwrite').parquet(TFT_static_dir)
        decoder_importance.write.mode('overwrite').parquet(TFT_decoder_dir)
        attention.write.mode('overwrite').parquet(TFT_attention_dir)
    else:
        print(f"appending new forecast values to the existing parquet {TFT_forcast_dir}")
        forecasts.write.mode('overwrite').parquet(TFT_forcast_dir)
        static_importance.write.mode('overwrite').parquet(TFT_static_dir)
        decoder_importance.write.mode('overwrite').parquet(TFT_decoder_dir)
        attention.write.mode('overwrite').parquet(TFT_attention_dir)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
