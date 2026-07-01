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

# CELL ********************

T_DRV_GRP_COLS = ['Indicator']
M_DRV_GRP_COLS = ['Region','Indicator']

T_ACT_GRP_COLS = ['Product_Category','series']
M_ACT_GRP_COLS = ['Product_Category', 'Region','series']

T_DRV_RENAME = {"Date":"feature_date","residual":"feature_residual"}
M_DRV_RENAME = {"Date":"feature_date","residual":"feature_residual", "Region":"feature_region"}
T_ACT_RENAME = {"Date":"target_date","residual":"target_residual"}
M_ACT_RENAME = {"Date":"target_date","residual":"target_residual","Region":"target_region"}

T_DRV_COLS_RN = ['Indicator']
T_ACT_COLS_RN = ['Product_Category', 'series']

M_DRV_COLS_RN = ['feature_region','Indicator']
M_ACT_COLS_RN = ['Product_Category', 'series', 'target_region']

T_series = ['series']
M_series = ['series']


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### ADF Test

# CELL ********************

compiled_drivers = spark.read.table("Sales_Forecasting.silver.compiled_drivers").select("Country","Indicator","Region","Date","Value")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

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

# CELL ********************


T_adf_schema = StructType(
    [StructField(c, StringType(), False) for c in T_DRV_GRP_COLS] +
    [
        StructField("adf_stat", DoubleType(), True),
        StructField("adf_p_value", DoubleType(), True),
        StructField("adf_stationary_flag", StringType(), True)
    ]
)

T_drivers = compiled_drivers.groupBy(*T_DRV_GRP_COLS, 'Date').agg(sum('Value').alias('Value'))
T_adf_results = T_drivers.groupBy(*T_DRV_GRP_COLS).applyInPandas(adf, T_adf_schema)




M_adf_schema = StructType(
    [StructField(c, StringType(), False) for c in M_DRV_GRP_COLS] +
    [
        StructField("adf_stat", DoubleType(), True),
        StructField("adf_p_value", DoubleType(), True),
        StructField("adf_stationary_flag", StringType(), True)       
    ]
)

M_drivers = compiled_drivers.groupBy(*M_DRV_GRP_COLS, 'Date').agg(sum('Value').alias('Value'))
M_adf_results = M_drivers.groupBy(*M_DRV_GRP_COLS).applyInPandas(adf, M_adf_schema)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(T_adf_results.groupBy('adf_stationary_flag').count())
display(M_adf_results.groupBy('adf_stationary_flag').count())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# def adf_group(df):
#     pdf = df.sort_values("Date")

#     country = pdf["Country"].iloc[0]
#     indicator = pdf["Indicator"].iloc[0]
#     region = pdf["Region"].iloc[0]

#     ts = (
#         pd.to_numeric(pdf["Value"], errors="coerce")
#         .replace([np.inf, -np.inf], np.nan)
#         .dropna()
#         .values
#     )


#     if len(ts) < 12:
#         return pd.DataFrame([{
#             "Country": country,
#             "Indicator": indicator,
#             "Region": region,
#             "adf_stat": None,
#             "adf_p_value": None,
#             "adf_stationary_flag": "Insufficient Data"            
#         }])
    
#     if np.nanstd(ts) == 0:
#         return pd.DataFrame([{
#             "Country": country,
#             "Indicator": indicator,
#             "Region": region,
#             "adf_stat": None,
#             "adf_p_value": None,
#             "adf_stationary_flag": "Constant Series (Skipped)"
#         }])


#     ## ADF execution
#     try:
#         adf_stat, p_value, *_ = adfuller(ts)

#         return pd.DataFrame([{
#             "Country": country,
#             "Indicator": indicator,
#             "Region": region,
#             "adf_stat": adf_stat,
#             "adf_p_value": p_value,
#             "adf_stationary_flag": (
#                 "Stationary" if p_value < 0.05
#                 else "Non Stationary"
#             )
#         }])

#     except Exception as e:
#         return pd.DataFrame([{
#             "Country": country,
#             "Indicator": indicator,
#             "Region": region,
#             "adf_stat": None,
#             "adf_p_value": None,
#             "adf_stationary_flag": f"ADF Failed: {str(e)}"
#         }])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# adf_topline_schema = StructType([
#     StructField("Country", StringType(), False),
#     StructField("Indicator", StringType(), False),
#     StructField("Region", StringType(), False),
#     StructField("adf_stat", DoubleType(), True),
#     StructField("adf_p_value", DoubleType(), True),
#     StructField("adf_stationary_flag", StringType(), True)
# ])

# adf_topline_results = (compiled_drivers
#                     .groupBy("Country","Indicator","Region")
#                     .applyInPandas(adf_group, schema=adf_schema)
# )


# adf_

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

# CELL ********************

T_kpss_schema = StructType(
    [StructField(c, StringType(), False) for c in T_DRV_GRP_COLS] +
    [
        StructField("kpss_stat", DoubleType(), True),
        StructField("kpss_p_value", DoubleType(), True),
        StructField("kpss_stationary_flag", StringType(), False)
    ]
)


T_kpss_results = T_drivers.groupBy(*T_DRV_GRP_COLS)\
                    .applyInPandas(kpss, schema=T_kpss_schema)


M_kpss_schema = StructType(
    [StructField(c, StringType(), False) for c in M_DRV_GRP_COLS] +
    [
        StructField("kpss_stat", DoubleType(), True),
        StructField("kpss_p_value", DoubleType(), True),
        StructField("kpss_stationary_flag", StringType(), False)
    ]
)

M_kpss_results = M_drivers.groupBy(*M_DRV_GRP_COLS)\
                    .applyInPandas(kpss, schema=M_kpss_schema)

        

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(T_kpss_results.groupBy('kpss_stationary_flag').count())
display(M_kpss_results.groupBy('kpss_stationary_flag').count())


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# def kpss_group(df):
#     pdf = df.sort_values("Date")

#     country = pdf["Country"].iloc[0]
#     indicator = pdf["Indicator"].iloc[0]
#     region = pdf["Region"].iloc[0]

#     # FORCE CLEAN NUMERIC PIPELINE
#     ts = (
#         pd.to_numeric(pdf["Value"], errors="coerce")
#         .replace([np.inf, -np.inf], np.nan)
#         .dropna()
#         .values
#     )


#     if len(ts) < 12:
#         return pd.DataFrame([{
#             "Country": country,
#             "Indicator": indicator,
#             "Region": region,
#             "kpss_stat": None,
#             "kpss_p_value": None,
#             "kpss_stationary_flag": "Insufficient Data (<12 obs)"
#         }])
    
#     # SAFETY CHECK 2: constant series
#     if np.nanstd(ts) == 0:
#         return pd.DataFrame([{
#             "Country": country,
#             "Indicator": indicator,
#             "Region": region,
#             "kpss_stat": None,
#             "kpss_p_value": None,
#             "kpss_stationary_flag": "Constant Series (Skipped)"
#         }])

        
#     ## Run KPSS
#     try:
#         kpss_stat, p_value, _, _ = kpss(
#             ts,
#             regression="c",
#             nlags="auto"
#         )

#         return pd.DataFrame([{
#                 "Country": country,
#                 "Indicator": indicator,
#                 "Region": region,
#                 "kpss_stat": kpss_stat,
#                 "kpss_p_value": p_value,
#                 "kpss_stationary_flag": 
#                     "Stationary" if p_value >= 0.05
#                     else "Non-Stationary"
#         }])


#     except Exception as e:
#         return pd.DataFrame([{
#                 "Country": country,
#                 "Indicator": indicator,
#                 "Region": region,
#                 "kpss_stat": None,
#                 "kpss_p_value": None,
#                 "kpss_stationary_flag": f"KPSS Failed: {str(e)}"
#         }])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# kpss_schema = StructType([
#                     StructField("Country", StringType(), False),
#                     StructField("Indicator", StringType(), False),
#                     StructField("Region", StringType(), False),
#                     StructField("kpss_stat", DoubleType(), True),
#                     StructField("kpss_p_value", DoubleType(), True),
#                     StructField("kpss_stationary_flag", StringType(), False)
# ])

# kpss_results = (compiled_drivers.groupBy("Country", "Indicator","Region")
#                     .applyInPandas(kpss_group, schema=kpss_schema))

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

# CELL ********************

T_results = T_adf_results.join(T_kpss_results, [*T_DRV_GRP_COLS], "inner")
M_results = M_adf_results.join(M_kpss_results, [*M_DRV_GRP_COLS], "inner")

T_stationary_stats = stationary_flag(T_results)
M_stationary_stats = stationary_flag(M_results)

display(T_stationary_stats.limit(10))
display(M_stationary_stats.limit(10))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

M_stationary_df = M_stationary_stats.toPandas()
T_stationary_df = T_stationary_stats.toPandas()

output_path = "/lakehouse/default/Files/Driver_Analysis/stationary_stats.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    T_stationary_df.to_excel(writer, sheet_name='topline_stats', index=False)
    M_stationary_df.to_excel(writer, sheet_name='middle_stats', index=False)
    print(f"saved to {output_path}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# driver_final_stats.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.driver_stats")

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

feature_set = spark.read.table("Sales_Forecasting.silver.compiled_drivers").select("Country","Indicator","Region","Date","Value")
topline_cutoff_data = spark.read.table("Sales_Forecasting.silver.topline_cutoff_data").withColumnRenamed("Quantity","Value")
middle_cutoff_data = spark.read.table("Sales_Forecasting.silver.middle_cutoff_data").withColumnRenamed("Quantity","Value")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# driver_classification = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.driver_stats")\
#         .drop('adf_stat',
#             'adf_p_value',
#             'adf_stationary_flag',
#             'kpss_stat',
#             'kpss_p_value',
#             'kpss_stationary_flag',)

# feature_set_valid = broadcast(
#     driver_classification.filter(col("valid_flag")==1)
# ).join(
#     feature_set, ["Country","Indicator","Region"], "inner"
# ).drop("valid_flag","stationary_class")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

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

# CELL ********************

T_schema = StructType(
    [StructField(c, StringType(), False) for c in T_ACT_GRP_COLS] +
    [
        StructField("Date", DateType(), False),
        StructField("Value", DoubleType(), True),
        StructField("residual", DoubleType(), True)
    ]
)

T_data = topline_cutoff_data.groupBy(*T_ACT_GRP_COLS,'Date').agg(sum('Value').alias('Value'))
topline_residuals = T_data.groupBy(*T_ACT_GRP_COLS)\
        .applyInPandas(
            stl_decompose,
            schema=T_schema
        )

M_schema = StructType(
    [StructField(c, StringType(), False) for c in M_ACT_GRP_COLS] +
    [
        StructField("Date", DateType(), False),
        StructField("Value", DoubleType(), True),
        StructField("residual", DoubleType(), True)
    ]
)

M_data = middle_cutoff_data.groupBy(*M_ACT_GRP_COLS,'Date').agg(sum("Value").alias("Value"))
middle_residuals = M_data.groupBy(*M_ACT_GRP_COLS)\
    .applyInPandas(stl_decompose, schema=M_schema)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

T_feature_set_schema = StructType(
    [StructField(c, StringType(), False) for c in T_DRV_GRP_COLS] +
    [
        StructField("Date", DateType(), False),
        StructField("Value", DoubleType(), True),
        StructField("residual", DoubleType(), True)
    ]
)

T_feature_set = feature_set.groupBy(*T_DRV_GRP_COLS,'Date').agg(sum('Value').alias('Value'))
T_feature_set_residuals = T_feature_set.groupBy(*T_DRV_GRP_COLS).applyInPandas(stl_decompose, schema=T_feature_set_schema)

M_feature_set_schema = StructType(
    [StructField(c, StringType(), False) for c in M_DRV_GRP_COLS] +
    [
        StructField("Date", DateType(), False),
        StructField("Value", DoubleType(), True),
        StructField("residual", DoubleType(), True)
    ]
)

M_feature_set = feature_set.groupBy(*M_DRV_GRP_COLS,'Date').agg(sum('Value').alias('Value'))
M_feature_set_residuals = M_feature_set.groupBy(*M_DRV_GRP_COLS).applyInPandas(stl_decompose, schema=M_feature_set_schema)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(topline_residuals.limit(10))
display(middle_residuals.limit(10))
display(T_feature_set_residuals.limit(10))
display(M_feature_set_residuals.limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Application of CCF
# 
# #### Joining target / drivers prior to CCF calculation

# CELL ********************

T_feature_set_residuals = T_feature_set_residuals.withColumnsRenamed(T_DRV_RENAME)
M_feature_set_residuals = M_feature_set_residuals.withColumnsRenamed(M_DRV_RENAME)
T_residuals = topline_residuals.withColumnsRenamed(T_ACT_RENAME)
M_residuals = middle_residuals.withColumnsRenamed(M_ACT_RENAME)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ##### Create mapping of features to targets

# CELL ********************

## CREATE MAPPING FOR TARGET-DRIVER COMBINATIONS

    ## TOPLINE
T_act_distinct = T_residuals.select(*T_ACT_COLS_RN).distinct()
T_drv_distinct = T_feature_set_residuals.select(*T_DRV_COLS_RN).distinct()

M_act_distinct = M_residuals.select(*M_ACT_COLS_RN).distinct()
M_drv_distinct = M_feature_set_residuals.select(*M_DRV_COLS_RN).distinct()

T_pairs = T_act_distinct.crossJoin(T_drv_distinct)
M_pairs = M_act_distinct.crossJoin(M_drv_distinct)

## also joining M_paris w/ the world level aggregated indicators from topline
M_w_pairs = M_act_distinct.crossJoin(T_drv_distinct)
M_w_pairs = M_w_pairs.withColumn("feature_region", lit('World'))

M_pairs = M_pairs.unionByName(M_w_pairs)


T_pairs_df = T_pairs.toPandas()
M_pairs_df = M_pairs.toPandas()

output_path = '/lakehouse/default/Files/Driver_Analysis/pairs.xlsx'

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    T_pairs_df.to_excel(writer, sheet_name='Topline_pairs', index=False)
    M_pairs_df.to_excel(writer, sheet_name='Middle_pairs', index=False)
    print(f"saved to {output_path}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

expanded_topline = T_residuals.join(broadcast(T_pairs), [*T_ACT_COLS_RN], 'inner')
expanded_middle = M_residuals.join(broadcast(M_pairs), [*M_ACT_COLS_RN], 'inner')


expanded_t_features = broadcast(T_pairs).join(T_feature_set_residuals, [*T_DRV_COLS_RN], 'inner')
expanded_m_features = broadcast(M_pairs).join(M_feature_set_residuals, [*M_DRV_COLS_RN], 'inner')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

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

# CELL ********************

topline_join_pairs = T_ACT_COLS_RN + T_DRV_COLS_RN
final_topline = join_target_feature(
    expanded_topline, expanded_t_features, topline_join_pairs)


middle_join_pairs = M_ACT_COLS_RN + M_DRV_COLS_RN
final_middle = join_target_feature(
    expanded_middle, expanded_m_features, middle_join_pairs)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

output_dir = "abfss://991f5e4b-c174-4ff2-992e-feb17d49d25a@onelake.dfs.fabric.microsoft.com/22746de3-183e-4327-a844-dceda0b7165c/Files/Driver_Analysis"

final_topline.write.mode("overwrite").parquet(
    f"{output_dir}/final_topline.parquet"
)

final_middle.write.mode("overwrite").parquet(
    f"{output_dir}/final_middle.parquet"
)

print("Parquet files written successfully.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# feature_set_residuals = feature_set_residuals\
#     .drop("Value")\
#     .withColumnsRenamed({"Date":"feature_date", "residual":"feature_residual","Region":"feature_region"})

# topline_residuals = topline_residuals\
#     .drop("Value")\
#     .withColumnsRenamed({"Date":"target_date","residual":"target_residual","Region":"target_region"})
    
# middle_residuals = middle_residuals\
#     .drop("Value")\
#     .withColumnsRenamed({"Date":"target_date","residual":"target_residual","Region":"target_region"})


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ## Create valid mapping for target-driver combinations

# ## create distinct records of each df set
# topline_distinct = topline_residuals.select("series").distinct()

# middle_distinct = middle_residuals.select("series","target_region").distinct()

# feature_world_distinct = feature_set_residuals.filter(col("feature_region")=="World").select("feature_region","Country","Indicator").distinct()

# ## Update to filter where Country=Region as well to make sure only to keep the aggregated versions of drivers for middle
# feature_m_distinct = feature_set_residuals.filter((~(col("feature_region")=="World")) & (col("feature_region")==col("Country")))\
#     .select("feature_region","Country","Indicator").distinct()


# ## create pair dataframes
# topline_pairs = broadcast(topline_distinct).crossJoin(feature_world_distinct)



# middle_w_pairs = broadcast(middle_distinct).crossJoin(feature_world_distinct)
# middle_pairs = broadcast(middle_distinct).join(
#     feature_m_distinct,
#     middle_distinct["target_region"]==feature_m_distinct["feature_region"],
#     "inner"
# )

# middle_pairs = middle_pairs.unionByName(middle_w_pairs)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# topline_pairs.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.topline_pairs")
# middle_pairs.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.middle_pairs")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ## Creating Expanded versions of the middle & topline residuals based on the pair mappings
#     ## both approaches for managing schema/column explosion during join work
    
# expanded_topline = topline_residuals.join(
#     broadcast(topline_pairs),
#     #topline_residuals["series"]==topline_pairs["series"],
#     ["series"],
#     "inner"
# )#.drop(topline_pairs["series"])

# display(expanded_topline.orderBy('series','Country','Indicator',asc('target_date')).limit(5))

# expanded_middle = middle_residuals.join(
#     broadcast(middle_pairs),
#     (middle_residuals["series"]==middle_pairs["series"]) & 
#     (middle_residuals["target_region"] == middle_pairs["target_region"]),
#     "inner"
# ).drop(middle_pairs["series"],middle_pairs["target_region"])

# display(expanded_middle.orderBy('series','Country','Indicator',asc('target_date')).limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# expanded_topline.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.expanded_topline")
# expanded_middle.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.expanded_middle")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ## Creating Expanded version of the middle & topline features based on the pair mappings
# expanded_t_features = broadcast(topline_pairs).join(
#     feature_set_residuals,
#     ["feature_region","Country","Indicator"],
#     "inner"
# )

# display(expanded_t_features.limit(3))

# expanded_m_features = broadcast(middle_pairs).join(
#     feature_set_residuals,
#     ["feature_region","Indicator","Country"],
#     "inner"
# )

# display(expanded_m_features.limit(3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# expanded_t_features.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.expanded_t_features")
# expanded_m_features.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.expanded_m_features")

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

# ## joining the feature and target residuals based on the pair mapping built within both dataframes
# expanded_t_features = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.expanded_t_features")
# expanded_m_features = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.expanded_m_features")


# expanded_topline = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.expanded_topline")
# expanded_middle = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.expanded_middle")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# # expanded_topline = spark.createDataFrame(expanded_topline.rdd, expanded_topline.schema)
# # expanded_middle = spark.createDataFrame(expanded_middle.rdd, expanded_middle.schema)

# # expanded_t_features = spark.createDataFrame(expanded_t_features.rdd, expanded_t_features.schema)
# # expanded_m_features = spark.createDataFrame(expanded_m_features.rdd, expanded_m_features.schema)


# t = expanded_topline.alias("t")
# m = expanded_middle.alias("m")
# tf = expanded_t_features.alias("tf")
# mf = expanded_m_features.alias("mf")


# final_topline = t.join(
#     tf,
#     (t["series"] == tf["series"]) &
#     (t["feature_region"] == tf["feature_region"]) &
#     (t["Country"] == tf["Country"]) &
#     (t["Indicator"] == tf["Indicator"]) &
#     (t["target_date"] == tf["feature_date"]),
#     "left"    
# ).select(t["series"],t["Product_Category"],t["target_date"],t["target_residual"],t["feature_region"],t["Country"],
#         t["Indicator"],tf["feature_residual"])

# middle_final = m.join(
#     mf,
#     (m["series"]==mf["series"]) &
#     (m["target_region"] ==mf["target_region"]) &
#     (m["feature_region"]==mf["feature_region"]) &
#     (m["Country"]==mf["Country"]) &
#     (m["Indicator"]==mf["Indicator"]) &
#     (m["target_date"]==mf["feature_date"]),
#     "left"
# ).select(m["series"],m["Product_Category"],m["target_region"],m["target_date"],m["target_residual"],
#         mf["feature_region"],mf["Indicator"],mf["Country"],mf["feature_residual"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# final_topline.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.topline_w_features")
# middle_final.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.middle_w_features")

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

        results.append(row)

    return pd.DataFrame(results)



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

T_cols = T_ACT_COLS_RN + T_DRV_COLS_RN
T_ccf_schema = StructType(
    [StructField(c, StringType(), False) for c in T_cols] +
    [

        StructField("Lag", IntegerType(), False),
        StructField("Correlation", DoubleType(), True)
    ]
)

T_ccf = final_topline.groupBy(*T_cols).applyInPandas(apply_ccf, schema = T_ccf_schema)



M_cols = M_ACT_COLS_RN + M_DRV_COLS_RN
M_ccf_schema = StructType(
    [StructField(c, StringType(), False) for c in M_cols] +
    [

        StructField("Lag", IntegerType(), False),
        StructField("Correlation", DoubleType(), True)
    ]
)

M_ccf = final_middle.groupBy(*M_cols).applyInPandas(apply_ccf, schema=M_ccf_schema)



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

base_dir = 'abfss://991f5e4b-c174-4ff2-992e-feb17d49d25a@onelake.dfs.fabric.microsoft.com/22746de3-183e-4327-a844-dceda0b7165c/Files/Driver_Analysis'

T_ccf.write.mode('overwrite').parquet(f"{base_dir}/topline_ccf_base.parquet")

M_ccf.write.mode('overwrite').parquet(f"{base_dir}/middle_ccf_base.parquet")

print("Parquet files written successfully.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# def compute_ccf(pdf):

#     pdf = pdf.sort_values("date")

#     driver = pdf["driver"]
#     target = pdf["target"]

#     results = []


#     for lag in range(-24, 24):

#         shifted_driver = driver.shift(lag)

#         valid = pd.concat([shifted_driver, target], axis=1,keys=['driver','target']).dropna()

#         if len(valid)>1:
#             corr = valid['driver'].corr(valid['target'])
#         else:
#             corr = None

#         if 'target_region' in pdf.columns:
#             results.append({
#                 "series": pdf["series"].iloc[0],
#                 "Product_Category": pdf["Product_Category"].iloc[0],
#                 "target_region": pdf["target_region"].iloc[0],
#                 "feature_region": pdf["feature_region"].iloc[0],
#                 "Country": pdf["Country"].iloc[0],
#                 "Indicator": pdf["Indicator"].iloc[0],
#                 "Lag": lag,
#                 "Correlation": (
#                                 float(0.0)
#                                 if corr is None or pd.isna(corr) or np.isinf(corr)
#                                 else float(corr)
#                             )
#             })

#         else:
#             results.append({
#                 "series": pdf["series"].iloc[0],
#                 "feature_region": pdf["feature_region"].iloc[0],
#                 "Country": pdf["Country"].iloc[0],
#                 "Indicator": pdf["Indicator"].iloc[0],
#                 "Lag": lag,
#                 "Correlation": (
#                                 float(0.0)
#                                 if corr is None or pd.isna(corr) or np.isinf(corr)
#                                 else float(corr)
#                             )
#             })

#     return pd.DataFrame(results)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# topline_w_features = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.topline_w_features")\
#     .withColumnsRenamed({"target_date":"date", "target_residual":"target", "feature_residual":"driver"})\
#     .select("series","Product_Category","feature_region","Country","Indicator","date","target","driver")

# middle_w_features = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.middle_w_features")\
#     .withColumnsRenamed({"target_date":"date", "target_residual":"target", "feature_residual":"driver"})\
#     .select("series","Product_Category","target_region","feature_region","Country","Indicator","date","target","driver")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# topline_ccf_schema = (StructType([
#     StructField("series", StringType(), False),
#     StructField("feature_region", StringType(), False),
#     StructField("Country", StringType(), False),
#     StructField("Indicator", StringType(), False),
#     StructField("Lag", IntegerType(), False),
#     StructField("Correlation", DoubleType(), True)
# ]))

# topline_ccf = topline_w_features.groupBy("series","feature_region","Country","Indicator")\
#     .applyInPandas(compute_ccf, schema=topline_ccf_schema)


# middle_ccf_schema = (StructType([
#     StructField("series", StringType(), False),
#     StructField("Product_Category", StringType(), True),
#     StructField("target_region", StringType(), True),
#     StructField("feature_region", StringType(), True),
#     StructField("Country", StringType(), True),
#     StructField("Indicator", StringType(), True),
#     StructField("Lag", IntegerType(), True),
#     StructField("Correlation", DoubleType(), True)
# ]))

# middle_ccf = middle_w_features.groupBy("series","Product_Category","target_region","feature_region","Country","Indicator")\
#     .applyInPandas(compute_ccf, schema=middle_ccf_schema)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# topline_ccf.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.topline_ccf_base")
# middle_ccf.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.middle_ccf_base")

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

base_dir = 'abfss://991f5e4b-c174-4ff2-992e-feb17d49d25a@onelake.dfs.fabric.microsoft.com/22746de3-183e-4327-a844-dceda0b7165c/Files/Driver_Analysis'

topline_ccf = spark.read.parquet(f"{base_dir}/topline_ccf_base.parquet")
middle_ccf = spark.read.parquet(f"{base_dir}/middle_ccf_base.parquet")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(topline_ccf.select(*T_cols).distinct().groupBy('series').count())
display(middle_ccf.select(*M_cols).distinct().groupBy('series').count())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

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
        (col("Indicator") != "NaN")
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

# CELL ********************

topline_ccf_filtered = ccf_filtering(topline_ccf, T_cols)
middle_ccf_filtered = ccf_filtering(middle_ccf, M_cols)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# topline_ccf = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.topline_ccf_base")
# middle_ccf = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.middle_ccf_base")

# display(topline_ccf.select('series','feature_region','Country','Indicator').distinct().groupBy('series').count())
# display(middle_ccf.select('series','feature_region','Country','Indicator').distinct().groupBy('series').count())



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# def ccf_filtering(df):
#     window = Window.partitionBy("series","feature_region","Country","Indicator")
#     df = df.withColumn("abs_corr", abs(col("Correlation")))
#     df = df.withColumn("max_corr", max(col("abs_corr")).over(window))


#     ## filter out records with a correlations < .3 or Indicator is NaN
#     df_filtered = df.filter(
#         (col("max_corr") > 0.15) &
#         # (col("max_corr") < .95) &
#         (col("Indicator").isNotNull()) &
#         (col("Indicator") != "NaN")
#     )

#     df_ranked = df_filtered.groupBy("series","feature_region","Country","Indicator").agg(first(col("max_corr")).alias("max_corr"))

#     w = Window.partitionBy("series").orderBy(desc("max_corr"))

#     df_ranked = df_ranked.withColumn("rank", rank().over(w))
#     df_rank_filtered = df_ranked.filter(col("rank")<=500)

#     df_final_filtered = df_rank_filtered.join(df_filtered, ['series','feature_region','Country','Indicator'], 'inner').drop(df_rank_filtered['max_corr'],df_rank_filtered['rank'])

#     return df_final_filtered

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# topline_ccf_filtered = ccf_filtering(topline_ccf)
# middle_ccf_filtered = ccf_filtering(middle_ccf)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# topline_ccf_filtered.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.topline_ccf_filtered") 
# middle_ccf_filtered.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.middle_ccf_filtered")

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

# CELL ********************

base_dir = 'abfss://991f5e4b-c174-4ff2-992e-feb17d49d25a@onelake.dfs.fabric.microsoft.com/22746de3-183e-4327-a844-dceda0b7165c/Files/Driver_Analysis'
middle_w_features = spark.read.parquet(f'{base_dir}/final_middle.parquet')
topline_w_features = spark.read.parquet(f'{base_dir}/final_topline.parquet')


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(topline_w_features.groupBy(*T_ACT_COLS_RN).agg(countDistinct(*T_DRV_COLS_RN).alias('count_indicators')))
display(middle_w_features.groupBy(*M_ACT_COLS_RN).agg(countDistinct(*M_DRV_COLS_RN).alias('count_indicators')))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

final_middle_features = top_x_feature_extraction(middle_w_features, middle_ccf_filtered, M_cols, M_ACT_COLS_RN, 30)

final_topline_features = top_x_feature_extraction(topline_w_features, topline_ccf_filtered, T_cols, T_ACT_COLS_RN, 30)

display(final_topline_features.limit(10))
display(final_middle_features.limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(final_topline_features.groupBy(*T_ACT_COLS_RN).agg(countDistinct(*T_DRV_COLS_RN).alias('count_indicators')))
display(final_middle_features.groupBy(*M_ACT_COLS_RN).agg(countDistinct(*M_DRV_COLS_RN).alias('count_indicators')))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_f = final_topline_features.orderBy(*T_ACT_COLS_RN, asc('feature_rank'), *T_DRV_COLS_RN, 'Lag').toPandas()
middle_f = final_middle_features.orderBy(*M_ACT_COLS_RN, asc('feature_rank'), *M_DRV_COLS_RN, asc('Lag')).toPandas()

output_path = "/lakehouse/default/Files/Driver_Analysis/feature_analysis.xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    topline_f.to_excel(writer, sheet_name="Topline_Features", index=False)
    middle_f.to_excel(writer, sheet_name="Middle_Featuers", index=False)
    print(f"saved to {output_path}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

final_topline_features.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.topline_top_features")
final_middle_features.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.middle_top_features")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# def corr_clustering(df):

#     serie = df['series'][0]
#     pivot_cols = ["series","target_date"]
#     middle_flag = False

#     if 'target_region' in df.columns:
#         pivot_cols.append("target_region")
#         target_region = df['target_region'][0]
#         middle_flag = True

#     wide_df = (
#         df.pivot_table(
#             index=pivot_cols,
#             columns="feature_id",
#             values="feature_residual",
#             aggfunc="first"
#             ).reset_index()
#         )

    
    


#     X = wide_df.drop(columns=pivot_cols)
#     X = X.select_dtypes(include=[np.number])
#     X = X.fillna(X.median(numeric_only=True))

#     corr = X.corr().abs()
#     corr = corr.fillna(0)
#     corr = np.clip(corr, 0, 1)

#     ## convert correlation matrix to distance matrix
#     distance = 1 - corr
#     distance = np.clip(distance, 0, 1)



#     ## hierarchical clustering 
#     condensed_dist = squareform(distance.values, checks=False)

#     linkage=sch.linkage(condensed_dist, method="average")

#     threshold = 0.2
#     cluster_labels = sch.fcluster(linkage,t=threshold, criterion="distance")
    
#     if middle_flag:
#         cluster_map = pd.DataFrame({
#             "serie": serie,
#             "target_region": target_region,
#             "feature_id": X.columns,
#             "cluster": cluster_labels
#         })
#     else:
#         cluster_map = pd.DataFrame({
#             "serie": serie,
#             "feature_id": X.columns,
#             "cluster": cluster_labels
#         })

#     representatives = []
#     for cluster_id in cluster_map["cluster"].unique():
#         features = cluster_map[cluster_map["cluster"] == cluster_id]["feature_id"].tolist()
        
#         if len(features) == 1:
#             representatives.append(features[0])
#             continue

#         sub_corr = corr.loc[features, features]

#         # score = mean correlation to others (higher = more central)
#         scores = sub_corr.mean(axis=1)

#         rep = scores.idxmax()
#         representatives.append(rep)

#     X_reduced = X[representatives]

#     final_df = pd.concat([df[["target_date"]], X_reduced], axis=1)


#     return cluster_map
    


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# def top_x_feature_extraction(df_f_resid, df_ccf_filtered, num_features):
#     ## creating a key column for feature/serie combinations
#     df_f_resid = df_f_resid.withColumn("feature_serie",
#         concat_ws("__", "series","feature_region","Country","Indicator"))


#     df_ccf_filtered = df_ccf_filtered.withColumn("feature_serie",
#         concat_ws("__", "series","feature_region","Country","Indicator"))
    
#     ## creating the feature_id for the feature/serie combinations that persisted after filtering
#     feature_series = df_ccf_filtered.select("feature_serie").distinct()
#     feature_series = feature_series.withColumn("feature_id", row_number().over(Window.orderBy("feature_serie")))

#     ## ensuring there is no ambiguity in the joins
#     r = df_f_resid.alias("r")


#     initial_joined_df = feature_series.join(df_ccf_filtered, ["feature_serie"], "left")
#     joined_df = initial_joined_df.join(df_f_resid, ['feature_serie'], 'inner')\
#         .drop(r["series"],r["feature_region"],r["Country"],r["Indicator"],r["feature_serie"])



#     ## updating logic to create the wide df later within the corr_clustering function
#     middle_process = False
#     if 'target_region' in joined_df.columns:
#         middle_process = True


#     if middle_process:
#         print("middle process")
#         middle_schema = StructType([
#             StructField("serie", StringType(), False),
#             StructField("target_region", StringType(), False),
#             StructField("feature_id", IntegerType(), False),
#             StructField("cluster", IntegerType(), False)
#         ])
#         clustering = joined_df.groupBy("series","target_region").applyInPandas(corr_clustering, schema=middle_schema)

#     else:
#         print("topline_process")
#         topline_schema = StructType([
#             StructField("serie", StringType(), False),
#             StructField("feature_id", IntegerType(), False),
#             StructField("cluster", IntegerType(), False)
#         ])

#         clustering = joined_df.groupBy("series").applyInPandas(corr_clustering, schema=topline_schema)



#     ij = initial_joined_df.alias("ij")
#     c = clustering.alias("c")

#     if middle_process:
#         data_w_cluster = ij.join(
#             c,
#             (ij['feature_id']==c['feature_id']) & (ij['series']==c['serie']) & (ij['target_region']==c['target_region']),
#             'inner'
#         ).drop(c['feature_id'],c['serie'],c['target_region'])

#     else:
#         data_w_cluster = ij.join(
#             c,
#             (ij['feature_id']==c['feature_id']) & (ij['series']==c['serie']),
#             'inner'
#         ).drop(c['feature_id'],c['serie'])

    

#     cluster_corr_w = Window.partitionBy("series", "cluster").orderBy(desc("max_corr"))

#     data_distinct = data_w_cluster.select("series","feature_id","cluster","max_corr").distinct()

#     data_distinct = data_distinct.withColumn("cluster_rank", row_number().over(cluster_corr_w))

#     data_filtered = data_distinct.filter(col("cluster_rank")<=3)


#     ## joining remaining features w/ original data
#     df = data_filtered.alias('df')

#     post_cluster_filtering = df.join(ij, ['feature_id'], 'inner').drop(df['series'],df['max_corr'])

#     ## ranking series within clusters
#     ranked_w = Window. partitionBy("series").orderBy(desc('max_corr'))

#     post_cluster_filtering = post_cluster_filtering.withColumn("feature_rank", dense_rank().over(ranked_w))

#     top_x_features = post_cluster_filtering.filter(col("feature_rank")<=num_features).select("series","feature_region","Country","Indicator","Lag","Correlation","abs_corr","max_corr","feature_rank")
#     top_x_features = top_x_features.withColumn("rec_lag", 
#         when(col("abs_corr")==col("max_corr"),lit(1)).otherwise(lit(0)))


#     top_x_features = top_x_features
#     w = Window.partitionBy("series","feature_region","Indicator").orderBy("Lag").rowsBetween(-1,1)
    
#     top_x_features = top_x_features\
#         .withColumn('rolling_corr_avg', mean(col("Correlation")).over(w))\
#         .withColumn('rolling_corr_std', stddev(col("Correlation")).over(w))\
#         .withColumn("stable_flag",
#             when(
#                 (col("Correlation") >= col("rolling_corr_avg") - col("rolling_corr_std")) &
#                 (col("Correlation") <= col("rolling_corr_avg") + col("rolling_corr_std")),
#                 lit(1)
#             ).otherwise(lit(0))
#             )


#     return top_x_features



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# middle_w_features = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.middle_w_features")\
#     .select("series","feature_region","Country","Indicator","target_date","feature_residual")
# topline_w_features = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.topline_w_features")\
#     .select("series","feature_region","Country","Indicator","target_date","feature_residual")

# middle_ccf_filtered = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.middle_ccf_filtered")
# topline_ccf_filtered = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.topline_ccf_filtered")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# final_middle_features = top_x_feature_extraction(middle_w_features, middle_ccf_filtered,30)

# final_topline_features = top_x_feature_extraction(topline_w_features, topline_ccf_filtered,30)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# display(final_topline_features.filter(col('series')=='ALU').select('Indicator').distinct())
# display(final_topline_features.filter(col('series')=='ALU'))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# final_topline_features.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.topline_top_features")
# final_middle_features.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.middle_top_features")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# topline_f = final_topline_features.orderBy("series","feature_region","feature_rank","Lag").toPandas()
# middle_f = final_middle_features.orderBy("series","feature_region","feature_rank","Lag").toPandas()

# output_path = "/lakehouse/default/Files/Driver_Analysis/feature_analysis.xlsx"
# with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
#     topline_f.to_excel(writer, sheet_name="Topline_Features", index=False)
#     middle_f.to_excel(writer, sheet_name="Middle_Featuers", index=False)
#     print(f"saved to {output_path}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Full parameterized pipeline / Horvath implementation

# CELL ********************

H_DRV_GRP_COLS = ['Indicator']
H_ACT_GRP_COLS = []

H_DRV_RENAME = {"Date":"feature_date","residual":"feature_residual"}
H_ACT_RENAME = {"Date":"target_date","residual":"target_residual"}

H_DRV_COLS_RN = ['Indicator']
H_ACT_COLS_RN = []

H_cols = (H_DRV_COLS_RN + H_ACT_COLS_RN)
H_series = []


## ADF test
H_adf_schema = StructType(
    [StructField(c, StringType(), False) for c in H_DRV_GRP_COLS] +
    [
        StructField("adf_stat", DoubleType(), True),
        StructField("adf_p_value", DoubleType(), True),
        StructField("adf_stationary_flag", StringType(), True)
    ]
)

H_drivers = compiled_drivers.groupBy(*H_DRV_GRP_COLS, 'Date').agg(sum('Value').alias('Value'))
H_adf_results = H_drivers.groupBy(*H_DRV_GRP_COLS).applyInPandas(adf, H_adf_schema)



## KPSS
H_kpss_schema = StructType(
    [StructField(c, StringType(), False) for c in H_DRV_GRP_COLS] +
    [
        StructField("kpss_stat", DoubleType(), True),
        StructField("kpss_p_value", DoubleType(), True),
        StructField("kpss_stationary_flag", StringType(), False)
    ]
)


H_kpss_results = H_drivers.groupBy(*H_DRV_GRP_COLS)\
                    .applyInPandas(kpss, schema=H_kpss_schema)



## COMBINING RESULSTS
H_results = H_adf_results.join(H_kpss_results, [*H_DRV_GRP_COLS], "inner")


H_stationary_display(H_stationary_stats.limit(10))



# original_data = spark.read.table("Sales_Forecasting.silver.topline_cutoff_data").withColumnRenamed("Quantity","Value")
# display(original_data.groupBy('series').agg(min('Date').alias('Date')).orderBy(asc('Date')))

## min date for cutoff was 2015-06-01 leverage for all data
H_org = spark.read.table('Sales_Forecasting.bronze.topline_data').withColumnRenamed("Quantity","Value")
H_org = H_org.filter(col("Date")>='2015-06-01')
H_data = H_org.groupBy('Date').agg(sum('Value').alias('Value'))
#H_data = H_data.withColumn('aggregation', lit('Horvath_topline'))

# horvath_schema = StructType([
#     StructField('aggregation', StringType(), False),
#     StructField('Date', DateType(), False),
#     StructField('Value', DoubleType(), True),
#     StructField('residual', DoubleType(), True)
# ])

# horvath_residuals = aggregated_data.groupBy('aggregation').applyInPandas(stl_decompose, schema=horvath_schema)

# display(horvath_residuals.limit(10))



## STL DECOMPOSITION

## H RESIDUALS
H_schema = StructType(
    [StructField(c, StringType(), False) for c in H_ACT_GRP_COLS] +
    [
        StructField("Date", DateType(), False),
        StructField("Value", DoubleType(), True),
        StructField("residual", DoubleType(), True)
    ]
)


H_residuals = H_data.groupBy(*H_ACT_GRP_COLS)\
        .applyInPandas(
            stl_decompose,
            schema=H_schema
        )



## FEATURE RESIDUALS
feature_set = spark.read.table("Sales_Forecasting.silver.compiled_drivers").select("Country","Indicator","Region","Date","Value")

H_feature_set_schema = StructType(
    [StructField(c, StringType(), False) for c in H_DRV_GRP_COLS] +
    [
        StructField("Date", DateType(), False),
        StructField("Value", DoubleType(), True),
        StructField("residual", DoubleType(), True)
    ]
)

H_feature_set = feature_set.groupBy(*H_DRV_GRP_COLS,'Date').agg(sum('Value').alias('Value'))
H_feature_set_residuals = H_feature_set.groupBy(*H_DRV_GRP_COLS).applyInPandas(stl_decompose, schema=H_feature_set_schema)




## APPLICATION OF CCF
H_feature_set_residuals = H_feature_set_residuals.withColumnsRenamed(H_DRV_RENAME)
H_residuals = H_residuals.withColumnsRenamed(H_ACT_RENAME)



## CREATE MAPPING FOR TARGET-DRIVER COMBINATIONS

    ## TOPLINE
H_act_distinct = H_residuals.select(*H_ACT_COLS_RN).distinct()
H_drv_distinct = H_feature_set_residuals.select(*H_DRV_COLS_RN).distinct()


H_pairs = H_act_distinct.crossJoin(H_drv_distinct)

if len(H_ACT_COLS_RN) == 0:
    H_expanded = H_residuals.crossJoin(broadcast(H_pairs))
else:
    H_expanded = H_residuals.join(broadcast(H_pairs), [*H_ACT_COLS_RN], 'inner')

H_expanded_features = broadcast(H_pairs).join(H_feature_set_residuals, [*H_DRV_COLS_RN], 'inner')


H_final = join_target_feature(
    H_expanded, H_expanded_features, H_cols
)



H_ccf_schema = StructType(
    [StructField(c, StringType(), False) for c in H_cols] +
    [

        StructField("Lag", IntegerType(), False),
        StructField("Correlation", DoubleType(), True)
    ]
)

H_ccf = H_final.groupBy(*H_cols).applyInPandas(apply_ccf, schema = H_ccf_schema)

H_ccf_filtered = ccf_filtering(H_ccf, H_cols, H_series)

H_w_features = H_final

display(H_w_features.groupBy(*H_ACT_COLS_RN).agg(countDistinct(*H_DRV_COLS_RN).alias('count_indicators')))

final_H_features = top_x_feature_extraction(H_w_features, H_ccf_filtered, H_cols, H_ACT_COLS_RN, 30)

display(final_H_features.groupBy(*H_ACT_COLS_RN).agg(countDistinct(*H_DRV_COLS_RN).alias("count_indicators")))


final_H = final_H_features.orderBy(*H_ACT_COLS_RN, asc('feature_rank'), *H_DRV_COLS_RN, 'Lag').toPandas()

output_path = "/lakehouse/default/Files/Driver_Analysis/Horvath_topline_feature_analysis.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    final_H.to_excel(writer, sheet_name="Horvath_Topline_Features", index=False)
    print(f"saved to {output_path}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 5. Complete Economic Validation
# - validate choices with Dominik for feature selection
# - Save the target time series / feature selection as output

# CELL ********************

topline_feature_selection = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.topline_top_features")
topline_feature_selection = topline_feature_selection.withColumn("feature_serie", concat_ws("__","feature_region","Country","Indicator"))
print(topline_feature_selection.columns)
display(topline_feature_selection.select('series').distinct())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(topline_feature_selection.filter(col("series")=="SCREWS").select("Indicator").distinct())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(topline_feature_selection.filter(col("series")=="SCREWS"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

middle_feature_selection = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.middle_top_features")
middle_feature_selection = middle_feature_selection.withColumn("feature_serie", concat_ws("__","feature_region","Country","Indicator"))
print(middle_feature_selection.columns)
display(middle_feature_selection.select('series').distinct().orderBy("series"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(middle_feature_selection.filter(col("series")=='PISTON___APAC'))

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

# ## Feature Expansion / Engineering
# For each Driver create the following using original data
# - Rolling (3, 6, 12) std, mean
# - Growth (YoY, MoM, diff)
# 
#         input: filtered drivers, raw data
#         output: engineered features

# MARKDOWN ********************

# ## Need to filter w/ the filtered/chosen features

# CELL ********************

## Feature Transformations to do
    ## levels
    ## Rolling Mean / STD (3,6,12)
    ## Growth: YoY, MoM, diff

## levels
drivers_FE = compiled_drivers.withColumn("level", col("Value"))

## Window creation
w = Window.partitionBy("Region", "Country","Indicator").orderBy("Date")
w_3 = Window.partitionBy("Region", "Country","Indicator").orderBy("Date").rowsBetween(-2,0)
w_6 = Window.partitionBy("Region", "Country", "Indicator").orderBy("Date").rowsBetween(-5,0)
w_12 = Window.partitionBy("Region", "Country", "Indicator").orderBy("Date").rowsBetween(-11,0)

## Rolling Means
drivers_FE = drivers_FE.withColumn("rolling_mean_3", avg("Value").over(w_3))
drivers_FE = drivers_FE.withColumn("rolling_mean_6", avg("Value").over(w_6))
drivers_FE = drivers_FE.withColumn("rolling_mean_12", avg("Value").over(w_12))

## Rolling STD
drivers_FE = drivers_FE.withColumn("rolling_std_3", stddev("Value").over(w_3))
drivers_FE = drivers_FE.withColumn("rolling_std_6", stddev("Value").over(w_6))
drivers_FE = drivers_FE.withColumn("rolling_std_12", stddev("Value").over(w_12))


## Growth: YoY, MoM, Diff
drivers_FE = drivers_FE.withColumn("diff", (col("Value") - lag("Value", 1).over(w)))
drivers_FE = drivers_FE.withColumn("YoY_pct", 
                        when(lag("Value",12).over(w) == 0, None)
                        .otherwise((col("Value")/lag("Value",12).over(w)) -1))
drivers_FE = drivers_FE.withColumn("MoM_pct",
                        when(lag("Value",1).over(w)==0, None)
                        .otherwise((col("Value") / lag("Value", 1).over(w))-1))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Full Pipeline Automation
# - ADF test
# - 

# CELL ********************

def adf(df):
    pdf = df.sort_values("Date")
    result = {c: pdf[c].iloc[0] for c in drv_grp_cols}


    ts = (
        pd.to_numeric(pdf["Value"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .values
    )


    if len(ts) < 12:
        return pd.DataFrame([{
            "adf_stat": None,
            "adf_p_value": None,
            "adf_stationary_flag": "Insufficient Data"            
        }])
    
    if np.nanstd(ts) == 0:
        result.update([{
            "adf_stat": None,
            "adf_p_value": None,
            "adf_stationary_flag": "Constant Series (Skipped)"
        }])
        return pd.DataFrame([result])


    ## ADF execution
    try:
        adf_stat, p_value, *_ = adfuller(ts)

        result.update([{
            "adf_stat": adf_stat,
            "adf_p_value": p_value,
            "adf_stationary_flag": (
                "Stationary" if p_value < 0.05
                else "Non Stationary"
            )
        }])
        return pd.DataFrame([result])

    except Exception as e:
        result.update([{
            "adf_stat": None,
            "adf_p_value": None,
            "adf_stationary_flag": f"ADF Failed: {str(e)}"
        }])
        return pd.DataFrame([result])
    

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def full_pipeline(compiled_drivers, drv_grp_cols ):
    adf_schema = StructType(
        [StructField(c, StringType(), False) for c in drv_grp_cols] +
        [
            StructField("adf_stat", DoubleType(), True),
            StructField("adf_p_value", DoubleType(), True),
            StructField("adf_stationary_flag", StringType(), True)
        ]
    )

    adf_results = compiled_drivers.groupBy(*drv_grp_cols).applyInPandas(adf, adf_schema)
    return adf_results

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

compiled_drivers = spark.read.table('Sales_Forecasting.silver.compiled_drivers')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

drv_grp_cols = ['Indicator']
topline_test = full_pipeline(compiled_drivers, drv_grp_cols)
display(topline_test)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def full_pipeline(compiled_drivers, drv_grp_cols):

    adf_schema = StructType(
        [StructField(c, StringType(), False) for c in drv_grp_cols] +
        [
            StructField("adf_stat", DoubleType(), True),
            StructField("adf_p_value", DoubleType(), True),
            StructField("adf_stationary_flag", StringType(), True)
        ]
    )

    def adf(df):

        pdf = df.sort_values("Date")

        # ALWAYS include group keys
        result = {c: pdf[c].iloc[0] for c in drv_grp_cols}

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

    return (
        compiled_drivers
        .groupBy(*drv_grp_cols)
        .applyInPandas(adf, schema=adf_schema)
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
