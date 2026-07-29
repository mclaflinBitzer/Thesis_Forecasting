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

# %pip install lightning

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# %pip install pytorch_forecasting

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Welcome to your new notebook
import numpy as np
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

# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import EarlyStopping

## old imports when using the pip install commented out above:
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ==========================================================
# CONFIG
# ==========================================================
manual_features = "/lakehouse/default/Files/Driver_Analysis/Final_Feature_Selection/"
automated_features = "/lakehouse/default/Files/Automated_Driver_Analysis/"
parquet_dir = "abfss://991f5e4b-c174-4ff2-992e-feb17d49d25a@onelake.dfs.fabric.microsoft.com/22746de3-183e-4327-a844-dceda0b7165c/Files/Forecasting"
Topline = True
rerun_historical_forecasts = False
driver_status = "Manual_Drivers"    ## options: "No_Drivers", "Manual_Drivers", "Automated_Drivers" 
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
    TFT_dir = parquet_dir + "TFT_Output.parquet"
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
    TFT_dir = parquet_dir + "TFT_Output.parquet"
FORECAST_HORIZON = 18
SEASONAL_PERIODS = 12
MIN_TRAIN = 36
STEP_SIZE = 3  # controls how many full model retrainings happen in the
               # walk-forward backtest below. Increase if too slow —
               # does not affect the always-on future forecast.

               
TUNE_HOLDOUT_MONTHS = FORECAST_HORIZON
N_OPTUNA_TRIALS =  2 #10 temp placement just for testing/verification
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
    print("Another trail/objective has been run")

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=5,
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
    train_loader = train_dataset.to_dataloader(
        train=True,
        batch_size=params["batch_size"],
        num_workers=10,
    )

    val_loader = validation_dataset.to_dataloader(
        train=False,
        batch_size=params["batch_size"],
        num_workers=10,
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
        max_epochs=15,
        limit_train_batches=0.8,
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
        train_loader,
        val_loader,
    )

    ########################################################
    # Evaluate
    ## this is just extracting the val_loss metric from the trainer
    ## how well did this trained model perform on unseen data?
    ########################################################
    print("Beginning model evaluation")
    score = trainer.validate(
        model,
        val_loader,
        verbose=False,
    )[0]["val_loss"]

    print('returning the score of the trail/run')
    print(f'score was {score}')
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
    validation_df = training_df[training_df.time_idx > cutoff-encoder_length]
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
            n_startup_trials=3,
            n_warmup_steps=5,
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

    display(sdf.orderBy(desc('Date')))

    training_df = sdf.filter(col(target_col).isNotNull())
    training_df = training_df.fillna(0.0, subset=driver_cols)
    display(training_df.orderBy(desc('Date')))

    training_df = training_df.toPandas()

    time_varying_known_reals = driver_cols + calendar_cols
    time_varying_known_reals.remove("_dt")

    sdf=sdf.toPandas()



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


    full_dataset = TimeSeriesDataSet(
        training_df,
        time_idx="time_idx",
        target=target_col,
        group_ids=ACT_GRP_COLS,
        max_encoder_length=ENCODER_LENGTH,
        max_prediction_length=FORECAST_HORIZON,
        static_categoricals=ACT_GRP_COLS,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=[target_col],
        allow_missing_timesteps=True,
    )

    train_loader = full_dataset.to_dataloader(
        train=True,
        batch_size=best_params["batch_size"],
    )

    model = TemporalFusionTransformer.from_dataset(
        full_dataset,
        learning_rate=best_params["learning_rate"],
        hidden_size=best_params["hidden_size"],
        attention_head_size=best_params["attention_head_size"],
        hidden_continuous_size=best_params["hidden_continuous_size"],
        dropout=best_params["dropout"],
        loss=QuantileLoss(),
    )

    trainer = pl.Trainer(
        max_epochs=50,
        gradient_clip_val=best_params["gradient_clip_val"],
        enable_progress_bar=False
    )

    trainer.fit(
        model,
        train_loader,
    )

    ## Creating Future DataSet / Predictions
    future_dataset = TimeSeriesDataSet.from_dataset(
        full_dataset,
        sdf,
        predict=True,
        stop_randomization=True,
    )

    future_loader = future_dataset.to_dataloader(
        train=False,
        batch_size=best_params['batch_size'],
    )

    predictions = trainer.predict(
        model,
        future_loader,
    )

    ## CODE TO EXTRACT IMPORTANCE METRICS
    interpretation = model.interpret_output(
        predictions.output,
        reduction="sum",
    )

    ## TFT does not produce static coefficients like classical models. it instead has dynamic importance 
        # scores that change for every forecast, target series, month, and forecast horizon


    ## THIS WILL CREATE 4 PLOTS
        # Static Variable Importance (ACT_GRP_COLS)
                # feature sthat never change within a series
                # this would output the importance of each of the ACT_GRP_COLS for the specific forecast scenario 
                # (ie the region was more important than product category)
        # Encoder Variable Importance
                # variables observed before the forecast begins and their weight
                # provides the importance of the input variables (includes driver_cols as well as historical target_col)
        # Decoder Variable Importance
                # variables known in advance during forecasting (most likely use this metric)
                # which future information mattered most when producing forecast values
        # Attention Weights
                # this explains which historical time periods mattered 
                # attention is not about variables/drivers it is instead highlighting which historical time periods the model looked at
                # ie period with the highest attention value means that the values from this historical period had the most influence on this particular forecast 
    model.plot_interpretation(interpretation)

    return predictions

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

sdf = actuals_fh_populated


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

display(sdf.orderBy(desc('Date')))

training_df = sdf.filter(col(target_col).isNotNull())
training_df = training_df.fillna(0.0, subset=driver_cols)
display(training_df.orderBy(desc('Date')))

training_df = training_df.toPandas()

time_varying_known_reals = driver_cols + calendar_cols
time_varying_known_reals.remove("_dt")


sdf=sdf.fillna(0.0, subset=driver_cols)
## need to also fill the target_col with 0 prior to ingestion for predictions as NaN/Nulls are not allowed
sdf=sdf.fillna(0.0, subset=target_col)
sdf=sdf.toPandas()


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


full_dataset = TimeSeriesDataSet(
    training_df,
    time_idx="time_idx",
    target=target_col,
    group_ids=ACT_GRP_COLS,
    max_encoder_length=ENCODER_LENGTH,
    max_prediction_length=FORECAST_HORIZON,
    static_categoricals=ACT_GRP_COLS,
    time_varying_known_reals=time_varying_known_reals,
    time_varying_unknown_reals=[target_col],
    allow_missing_timesteps=True,
)

train_loader = full_dataset.to_dataloader(
    train=True,
    batch_size=best_params["batch_size"],
)

model = TemporalFusionTransformer.from_dataset(
    full_dataset,
    learning_rate=best_params["learning_rate"],
    hidden_size=best_params["hidden_size"],
    attention_head_size=best_params["attention_head_size"],
    hidden_continuous_size=best_params["hidden_continuous_size"],
    dropout=best_params["dropout"],
    loss=QuantileLoss(),
)

trainer = pl.Trainer(
    max_epochs=50,
    gradient_clip_val=best_params["gradient_clip_val"],
    enable_progress_bar=False
)

trainer.fit(
    model,
    train_loader,
)

## Creating Future DataSet / Predictions
future_dataset = TimeSeriesDataSet.from_dataset(
    full_dataset,
    sdf,
    predict=True,
    stop_randomization=True,
)

future_loader = future_dataset.to_dataloader(
    train=False,
    batch_size=best_params['batch_size'],
)

predictions = trainer.predict(
    model,
    future_loader,
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

future_check = sdf.copy()
future_check.isna().sum().sort_values(ascending=False)

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
