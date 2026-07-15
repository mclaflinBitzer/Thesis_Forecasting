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
from pyspark.sql.functions import col
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.types import *
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from pyspark.sql.utils import AnalysisException
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import ccf 
import statsmodels.api as sm
from statsmodels.tsa.stattools import kpss as kpss_test
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from matplotlib.ticker import PercentFormatter
from pyspark.sql.functions import pandas_udf
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from functools import reduce
import operator
from pyspark.sql.functions import col

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### ADF Test

# CELL ********************

def adf(df):

    pdf = df.sort_values("Date")

    # ALWAYS include group keys
    grp_cols = [c for c in pdf.columns if c not in ['Date','Value']]
    result = {c: pdf[c].iloc[0] for c in grp_cols}

    ts = (
        pd.to_numeric(pdf["Value"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .values
    )

    # -------------------------
    # insufficient data
    # -------------------------
    if len(ts) < 12:
        result.update({
            "adf_stat": None,
            "adf_p_value": None,
            "adf_stationary_flag": "Insufficient Data"
        })
        return pd.DataFrame([result])

    # -------------------------
    # constant series
    # -------------------------
    if np.nanstd(ts) == 0:
        result.update({
            "adf_stat": None,
            "adf_p_value": None,
            "adf_stationary_flag": "Constant Series (Skipped)"
        })
        return pd.DataFrame([result])

    # -------------------------
    # ADF test
    # -------------------------
    try:
        adf_stat, p_value, *_ = adfuller(ts)

        result.update({
            "adf_stat": adf_stat,
            "adf_p_value": p_value,
            "adf_stationary_flag": (
                "Stationary" if p_value < 0.05 else "Non Stationary"
            )
        })

    except Exception as e:
        result.update({
            "adf_stat": None,
            "adf_p_value": None,
            "adf_stationary_flag": f"ADF Failed: {str(e)}"
        })

    return pd.DataFrame([result])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### KPSS Test

# CELL ********************

def kpss(df):
    pdf = df.sort_values('Date')

    # ALWAYS include group keys
    grp_cols = [c for c in pdf.columns if c not in ['Date','Value']]
    result = {c: pdf[c].iloc[0] for c in grp_cols}


    # FORCE CLEAN NUMERIC PIPELINE
    ts = (
        pd.to_numeric(pdf["Value"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .values
    )

    
    if len(ts) < 12:
        result.update({
            "kpss_stat": None,
            "kpss_p_value": None,
            "kpss_stationary_flag": "Insufficient Data (<12 obs)"
        })
        return pd.DataFrame([result])
    
    # SAFETY CHECK 2: constant series
    if np.nanstd(ts) == 0:
        result.update({
            "kpss_stat": None,
            "kpss_p_value": None,
            "kpss_stationary_flag": "Constant Series (Skipped)"
        })
        return pd.DataFrame([result])

        
    ## Run KPSS
    try:
        kpss_stat, p_value, _, _ = kpss_test(
            ts)

        result.update({
                "kpss_stat": kpss_stat,
                "kpss_p_value": p_value,
                "kpss_stationary_flag": 
                    "Stationary" if p_value >= 0.05
                    else "Non-Stationary"
        })
        return pd.DataFrame([result])


    except Exception as e:
        result.update({
                "kpss_stat": None,
                "kpss_p_value": None,
                "kpss_stationary_flag": f"KPSS Failed: {str(e)}"
        })
        return pd.DataFrame([result])


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Combine adf & kpss outputs
# - combine outputs
# - apply valid or invalid flag
# - write to driver stats table 
# - remove records where either adf or kpss has skipped, flagged for insufficient data, or the test has failed
# - create binary flag for stationary or requires differencing


# CELL ********************

def stationary_flag(df_stationary_stats):
        
    ## add valid/invalid flag
        ## if valid then 1 otherwise 0
    df_stationary_stats = df_stationary_stats.withColumn("valid_flag", 
        when(
            (
                (col("adf_stationary_flag").isin("Constant Series (Skipped)", "ADF Failed: Invalid input, x is constant", "Insufficient Data")) |
                (col("kpss_stationary_flag").isin("Constant Series (Skipped)", "Insufficient Data (<12 obs)","KPSS Failed: cannot convert float infinity to integer"))
            ), lit(0)
        ).otherwise(lit(1))
        
    )

    driver_final_stats = df_stationary_stats.withColumn(
        "ADF_is_stationary",
        when(col("valid_flag")==0, None)
        .when(col("adf_p_value") < 0.05, 1)
        .otherwise(0)
    )

    driver_final_stats = driver_final_stats.withColumn(
        "KPSS_is_stationary",
        when(col("valid_flag")==0, None)
        .when(col("kpss_p_value") > 0.05, 1)
        .otherwise(0)
    )


    return driver_final_stats
#df_stationary_stats = adf_results.join(kpss_results, ["Country", "Indicator","Region"])


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Stage 2: Target / Driver Transformations
# **Purpose:** Align target + driver values using STL decomposition
# ##### Transformations - Apply STL in order to only do the correlation analysis upon the residuals
# ---

# CELL ********************

def stl_decompose(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf = pdf.sort_values("Date")
    pdf["Value"] = pdf["Value"].replace(np.nan, 0)

    stl = STL(pdf["Value"], period=12, robust=True)
    result = stl.fit()

    #pdf["trend"] = result.trend
    #pdf["seasonal"] = result.seasonal
    pdf["residual"] = result.resid

    return pdf

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Application of CCF
# 
# #### Joining target / drivers prior to CCF calculation

# MARKDOWN ********************

# ##### Create mapping of features to targets

# CELL ********************

def join_target_feature(
    target_df,
    feature_df,
    join_pairs,
    target_alias="t",
    feature_alias="f",
):

    t = target_df.alias(target_alias)
    f = feature_df.alias(feature_alias)

    conditions = []

    for c in join_pairs:
        # Automatically map target_* -> feature_*
        feature_col = c

        conditions.append(
            col(f"{target_alias}.{c}") == col(f"{feature_alias}.{feature_col}")
        )


    # Always join on the dates
    conditions.append(
        col(f"{target_alias}.target_date") ==
        col(f"{feature_alias}.feature_date")
    )

    join_cond = reduce(operator.and_, conditions)

    return (
        t.join(f, join_cond, "left")
         .select(
             *[col(f"{target_alias}.{c}") for c in target_df.columns],
             col(f"{feature_alias}.feature_residual")
         )
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### CCF Compute

# CELL ********************

def apply_ccf(df):
    pdf = df.sort_values('target_date')

    result_cols = [c for c in pdf.columns if c not in ['target_date','Value','feature_residual','target_residual']]

    base_result = {c: pdf[c].iloc[0] for c in result_cols}


    driver = pd.to_numeric(pdf["feature_residual"], errors="coerce")
    target = pd.to_numeric(pdf["target_residual"], errors="coerce")

    results = []

    #total possible observations for the combination
    total_obs = len(pdf)

    for lag in range(-24, 25):      # include +24

        shifted_driver = driver.shift(lag)

        valid = (
            pd.concat(
                [shifted_driver, target],
                axis=1,
                keys=["driver", "target"]
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        n_overlap = len(valid)
        coverage = n_overlap / total_obs if total_obs>0 else 0.0

        if len(valid) > 1:
            corr = valid["driver"].corr(valid["target"])
        else:
            corr = np.nan

        row = base_result.copy()
        row["Lag"] = lag
        row["Correlation"] = (
            0.0
            if pd.isna(corr) or np.isinf(corr)
            else float(corr)
        )
        row['n_overlap'] = int(n_overlap)
        row['coverage'] = float(coverage)

        results.append(row)

    return pd.DataFrame(results)



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Stage 4: Feature Selection
# For each time series w/ all potential driver/feature combinations
# - Correlation Filtering (Pearson/Spearman) strength threshold/ranking
# - Feature Cluster filtering 
# 
#         input: raw driver data at lag/lead identified, engineered features at lag/lead identified 
#         output: take the top x features

# CELL ********************

def ccf_filtering(df, cols, serie_col):
    window = Window.partitionBy(*cols)
    df = df.withColumn("abs_corr", abs(col("Correlation")))
    df = df.withColumn("max_corr", max(col("abs_corr")).over(window))


    ## filter out records with a correlations < .3 or Indicator is NaN
    df_filtered = df.filter(
        (col("max_corr") > 0.15) &
        # (col("max_corr") < .95) &
        (col("Indicator").isNotNull()) &
        (col("Indicator") != "NaN") &
        (col('coverage') >= 0.8) &
        (col('n_overlap') >= 24)
    )

    df_ranked = df_filtered.groupBy(*cols).agg(first(col("max_corr")).alias("max_corr"))

    if len(serie_col) == 0:
        w = Window.orderBy(desc('max_corr'))
    else:
        w = Window.partitionBy(*serie_col).orderBy(desc("max_corr"))

    df_ranked = df_ranked.withColumn("rank", rank().over(w))
    df_rank_filtered = df_ranked.filter(col("rank")<=500)

    df_final_filtered = df_rank_filtered.join(df_filtered, [*cols], 'inner').drop(df_rank_filtered['max_corr'],df_rank_filtered['rank'])

    return df_final_filtered

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Correlation Clustering

# CELL ********************

def corr_clustering(df):

    # grouping columns are whatever isn't a measure
    group_cols = [
        c for c in df.columns
        if c not in [
            "target_date",
            "feature_id",
            "feature_residual"
        ]
    ]

    group_values = {
        c: df[c].iloc[0]
        for c in group_cols
    }

    wide_df = (
        df.pivot_table(
            index=group_cols + ["target_date"],
            columns="feature_id",
            values="feature_residual",
            aggfunc="first"
        )
        .reset_index()
    )

    X = (
        wide_df
        .drop(columns=group_cols + ["target_date"])
        .select_dtypes(include=[np.number])
        .fillna(0)
    )

    corr = X.corr().abs().fillna(0)
    distance = 1 - corr

    linkage = sch.linkage(
        squareform(distance.values, checks=False),
        method="average"
    )

    labels = sch.fcluster(
        linkage,
        t=0.2,
        criterion="distance"
    )

    cluster_map = pd.DataFrame({
        **group_values,
        "feature_id": X.columns,
        "cluster": labels
    })

    return cluster_map

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def top_x_feature_extraction(df_f_resid, df_ccf_filtered, grp_cols, act_cols, num_features):
    df_f_resid = df_f_resid.withColumn("feature_serie", concat_ws("__", *grp_cols))

    df_ccf_filtered = df_ccf_filtered.withColumn("feature_serie", concat_ws("__", *grp_cols))


    ## creating the feature_id for the feature/serie combinations that persisted after filtering
    feature_series = df_ccf_filtered.select("feature_serie").distinct()
    feature_series = feature_series.withColumn("feature_id", row_number().over(Window.orderBy("feature_serie")))

    ## ensuring there is no ambiguity in the joins

    r = df_f_resid.alias("r")


    initial_joined_df = feature_series.join(df_ccf_filtered, ["feature_serie"], "left")
    joined_df = initial_joined_df.join(r, ['feature_serie'], 'inner')\
        .drop(*[col(f"r.{c}") for c in grp_cols])\
        .select(*grp_cols, "target_date", "feature_id", "feature_residual" )


    schema = StructType(
        [StructField(c, StringType(), False) for c in grp_cols] +
        [
            StructField("feature_id", IntegerType(), False),
            StructField("cluster", IntegerType(), False)
        ]
    )

    clustering = joined_df.groupBy(*act_cols).applyInPandas(corr_clustering, schema=schema)



    ij = initial_joined_df.alias("ij")
    c_df = clustering.alias("c")
        
    conditions = []

    for col_nam in act_cols:
        # Automatically map target_* -> feature_*
        conditions.append(
            col(f"ij.{col_nam}") == col(f"c.{col_nam}")
        )
    # Always join on the dates
    conditions.append(
        col(f"ij.feature_id") ==
        col(f"c.feature_id")
    )

    join_cond = reduce(operator.and_, conditions)

    data_w_cluster = ij.join(
        c_df,
        join_cond, 'inner'
    ).select(*[col(f"ij.{c}") for c in ij.columns],c_df['cluster'])

    if 'series' in grp_cols:
        cluster_corr_w = Window.partitionBy("series", "cluster").orderBy(desc("max_corr"))

        data_distinct = data_w_cluster.select("series","feature_id","cluster","max_corr").distinct()

        data_distinct = data_distinct.withColumn("cluster_rank", row_number().over(cluster_corr_w))

        data_filtered = data_distinct.filter(col("cluster_rank")<=3)

        ## joining remaining features w/ original data
        df = data_filtered.alias('df')

        post_cluster_filtering = df.join(ij, ['feature_id'], 'inner').drop(df['series'],df['max_corr'])

        ## ranking series within clusters
        ranked_w = Window. partitionBy("series").orderBy(desc('max_corr'))

        post_cluster_filtering = post_cluster_filtering.withColumn("feature_rank", dense_rank().over(ranked_w))

    else:
        cluster_corr_w = Window.partitionBy("cluster").orderBy(desc("max_corr"))

        data_distinct = data_w_cluster.select("feature_id","cluster","max_corr").distinct()

        data_distinct = data_distinct.withColumn("cluster_rank", row_number().over(cluster_corr_w))

        data_filtered = data_distinct.filter(col("cluster_rank")<=3)

        ## joining remaining features w/ original data
        df = data_filtered.alias('df')

        post_cluster_filtering = df.join(ij, ['feature_id'], 'inner').drop(df['max_corr'])

        ## ranking w/o series within clusters
        ranked_w = Window.orderBy(desc('max_corr'))

        post_cluster_filtering = post_cluster_filtering.withColumn("feature_rank", dense_rank().over(ranked_w))    

    top_x_features = post_cluster_filtering.filter(col("feature_rank")<=num_features).select(*grp_cols,"Lag","Correlation","abs_corr","max_corr","feature_rank")


    top_x_features = top_x_features.withColumn("rec_lag", 
        when(col("abs_corr")==col("max_corr"),lit(1)).otherwise(lit(0)))


    top_x_features = top_x_features
    w = Window.partitionBy(*grp_cols).orderBy("Lag").rowsBetween(-1,1)

    top_x_features = top_x_features\
        .withColumn('rolling_corr_avg', mean(col("Correlation")).over(w))\
        .withColumn('rolling_corr_std', stddev(col("Correlation")).over(w))\
        .withColumn("stable_flag",
            when(
                (col("Correlation") >= col("rolling_corr_avg") - col("rolling_corr_std")) &
                (col("Correlation") <= col("rolling_corr_avg") + col("rolling_corr_std")),
                lit(1)
            ).otherwise(lit(0))
            )



    return top_x_features


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Full parameterized pipeline / Horvath implementation

# CELL ********************

## BASE DIRECTORY FOR ALL OUTPUTS
parquet_base_dir = 'abfss://991f5e4b-c174-4ff2-992e-feb17d49d25a@onelake.dfs.fabric.microsoft.com/22746de3-183e-4327-a844-dceda0b7165c/Files/Driver_Analysis/'
excel_base_dir = "/lakehouse/default/Files/Driver_Analysis/"

## TOPLINE
T_actuals_table = "Sales_Forecasting.silver.topline_cutoff_data"

T_target_col = 'Quantity'

T_DRV_GRP_COLS = ['Indicator']
T_ACT_GRP_COLS = ['Product_Category','series']


T_DRV_RENAME = {"Date":"feature_date","residual":"feature_residual"}
T_ACT_RENAME = {"Date":"target_date","residual":"target_residual"}


T_num_features = 30

T_stationary_stats_output_path = excel_base_dir + "topline_stationary_stats.xlsx"
T_pairs_path = excel_base_dir + "topline_pairs.xlsx"
T_joined_act_f_dir = parquet_base_dir + "final_topline.parquet"
T_ccf_output_dir = parquet_base_dir + 'topline_ccf_base.parquet'
T_final_feature_dir = excel_base_dir + 'topline_recommended_features.xlsx'

## MIDDLE
M_actuals_table = "Sales_Forecasting.silver.middle_cutoff_data"

M_target_col = "Quantity"

M_DRV_GRP_COLS = ['Region','Indicator']
M_ACT_GRP_COLS = ['Product_Category', 'Region','series']

M_DRV_RENAME = {"Date":"feature_date","residual":"feature_residual"} 
M_ACT_RENAME = {"Date":"target_date","residual":"target_residual"}

M_num_features = 30

M_stationary_stats_output_path = excel_base_dir + "middle_stationary_stats.xlsx"
M_pairs_path = excel_base_dir + "middle_pairs.xlsx"
M_joined_act_f_dir = parquet_base_dir + "final_middle.parquet"
M_ccf_output_dir = parquet_base_dir + 'middle_ccf_base.parquet'
M_final_feature_dir = excel_base_dir + 'middle_recommended_features.xlsx'

Topline = False

if Topline:

    actuals_table = T_actuals_table 
    
    target_col = T_target_col

    DRV_GRP_COLS = T_DRV_GRP_COLS 
    ACT_GRP_COLS = T_ACT_GRP_COLS 


    ACT_RENAME = T_ACT_RENAME
    DRV_RENAME = T_DRV_RENAME

    num_features = T_num_features
    
    stationary_stats_output_path = T_stationary_stats_output_path
    pairs_path = T_pairs_path
    joined_act_f_dir = T_joined_act_f_dir
    ccf_output_dir = T_ccf_output_dir
    final_feature_dir = T_final_feature_dir


else:
    
    actuals_table = M_actuals_table

    target_col = M_target_col

    DRV_GRP_COLS = M_DRV_GRP_COLS 
    ACT_GRP_COLS = M_ACT_GRP_COLS 

    ACT_RENAME = M_ACT_RENAME
    DRV_RENAME = M_DRV_RENAME

    num_features = M_num_features

    stationary_stats_output_path = M_stationary_stats_output_path
    pairs_path = M_pairs_path
    joined_act_f_dir = M_joined_act_f_dir
    ccf_output_dir = M_ccf_output_dir
    final_feature_dir = M_final_feature_dir


## shared
JOIN_T_F_cols = list(dict.fromkeys(DRV_GRP_COLS + ACT_GRP_COLS))
col_renamed = {"Quantity":"target_value","Value":"feature_value"}
driver_table = "Sales_Forecasting.silver.compiled_drivers"
series_col = ['series']



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

compiled_drivers = spark.read.table("Sales_Forecasting.silver.compiled_drivers").select("Country","Indicator","Region","Date","Value")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

adf_schema = StructType(
    [StructField(c, StringType(), True) for c in DRV_GRP_COLS] +
    [
        StructField("adf_stat", DoubleType(), True),
        StructField("adf_p_value", DoubleType(), True),
        StructField("adf_stationary_flag", StringType(), True)        
    ]
)

drivers = compiled_drivers.groupBy(*DRV_GRP_COLS, 'Date').agg(sum('Value').alias('Value'))
adf_results = drivers.groupBy(*DRV_GRP_COLS).applyInPandas(adf, adf_schema)
display(adf_results.groupBy('adf_stationary_flag').count())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

kpss_schema = StructType(
    [StructField(c, StringType(), True) for c in DRV_GRP_COLS] +
    [
        StructField("kpss_stat", DoubleType(), True),
        StructField("kpss_p_value", DoubleType(), True),
        StructField("kpss_stationary_flag", StringType(), False)       
    ]
)

kpss_results = drivers.groupBy(DRV_GRP_COLS).applyInPandas(kpss, kpss_schema)
display(kpss_results.groupBy('kpss_stationary_flag').count())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

results = adf_results.join(kpss_results, [*DRV_GRP_COLS], 'inner')
stationary_stats = stationary_flag(results)

stationary_df = stationary_stats.toPandas()

stationary_df.to_excel(stationary_stats_output_path)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

feature_set = compiled_drivers
cutoff_data = spark.read.table(actuals_table).withColumnRenamed(target_col, 'Value')


act_resid_schema = StructType(
    [StructField(c, StringType(), False) for c in ACT_GRP_COLS] +
    [
        StructField("Date", DateType(), False),
        StructField("Value", DoubleType(), True),
        StructField("residual", DoubleType(), True)
    ]
)

data = cutoff_data.groupBy(*ACT_GRP_COLS,'Date').agg(sum('Value').alias('Value'))
act_residuals = data.groupBy(*ACT_GRP_COLS)\
        .applyInPandas(
            stl_decompose,
            schema=act_resid_schema
        )
display(act_residuals.limit(3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

feature_resid_schema = StructType(
    [StructField(c, StringType(), True) for c in DRV_GRP_COLS] +
    [
        StructField("Date", DateType(), False),
        StructField("Value", DoubleType(), True),
        StructField("residual", DoubleType(), True)
    ]
)

feature_set = feature_set.groupBy(*DRV_GRP_COLS,'Date').agg(sum('Value').alias('Value'))
feature_set_residuals = feature_set.groupBy(*DRV_GRP_COLS).applyInPandas(stl_decompose, schema=feature_resid_schema)

display(feature_set_residuals.limit(3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

feature_set_residuals = feature_set_residuals.withColumnsRenamed(DRV_RENAME)
act_residuals = act_residuals.withColumnsRenamed(ACT_RENAME)

act_distinct = act_residuals.select(*ACT_GRP_COLS).distinct()
drv_distinct = feature_set_residuals.select(*DRV_GRP_COLS).distinct()


join_col = []
for a_col in ACT_GRP_COLS:
    for d_col in DRV_GRP_COLS:
        if a_col == d_col:
            join_col.append(d_col)
            print(f"{d_col} added to join col list")


if len(join_col) == 0:
    print("no shared columns so cross join was done")
    pairs = act_distinct.crossJoin(drv_distinct)
else:
    print(f"shared columns so the join was done on {join_col}")
    pairs = act_distinct.join(drv_distinct, join_col, 'inner')


if len(ACT_GRP_COLS) == 0:
    expanded_act = act_residuals.crossJoin(broadcast(pairs))
else:
    expanded_act = act_residuals.join(broadcast(pairs), [*ACT_GRP_COLS], 'inner')

expanded_features = broadcast(pairs).join(feature_set_residuals, [*DRV_GRP_COLS], 'inner')

final = join_target_feature(
    expanded_act, expanded_features, JOIN_T_F_cols
)



pairs_pd = pairs.toPandas()
pairs_pd.to_excel(pairs_path)
final.write.mode('overwrite').parquet(joined_act_f_dir)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

ccf_schema = StructType(
    [StructField(c, StringType(), False) for c in JOIN_T_F_cols] +
    [

        StructField("Lag", IntegerType(), False),
        StructField("Correlation", DoubleType(), True),
        StructField("n_overlap", IntegerType(), False),
        StructField("coverage", DoubleType(), False)
    ]
)

ccf_output = final.groupBy(JOIN_T_F_cols).applyInPandas(apply_ccf, schema=ccf_schema)

ccf_output.write.mode('overwrite').parquet(ccf_output_dir)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

ccf_filtered = ccf_filtering(ccf_output, JOIN_T_F_cols, series_col)

display(ccf_output.groupBy(*ACT_GRP_COLS).agg(countDistinct(*DRV_GRP_COLS).alias('count_indicators')))

final_features = top_x_feature_extraction(final, ccf_filtered, JOIN_T_F_cols, ACT_GRP_COLS, num_features)

display(final_features.groupBy(*ACT_GRP_COLS).agg(countDistinct(*DRV_GRP_COLS).alias('count_indicators')))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

final_features_df = final_features.orderBy(*ACT_GRP_COLS, asc('feature_rank'), *DRV_GRP_COLS, 'Lag').toPandas()
final_features_df.to_excel(final_feature_dir)


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
