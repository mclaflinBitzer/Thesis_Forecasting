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
from statsmodels.tsa.stattools import kpss
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from matplotlib.ticker import PercentFormatter
from pyspark.sql.functions import pandas_udf
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

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

def adf_group(df):
    pdf = df.sort_values("Date")

    country = pdf["Country"].iloc[0]
    indicator = pdf["Indicator"].iloc[0]
    region = pdf["Region"].iloc[0]

    ts = (
        pd.to_numeric(pdf["Value"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .values
    )


    if len(ts) < 12:
        return pd.DataFrame([{
            "Country": country,
            "Indicator": indicator,
            "Region": region,
            "adf_stat": None,
            "adf_p_value": None,
            "adf_stationary_flag": "Insufficient Data"            
        }])
    
    if np.nanstd(ts) == 0:
        return pd.DataFrame([{
            "Country": country,
            "Indicator": indicator,
            "Region": region,
            "adf_stat": None,
            "adf_p_value": None,
            "adf_stationary_flag": "Constant Series (Skipped)"
        }])


    ## ADF execution
    try:
        adf_stat, p_value, *_ = adfuller(ts)

        return pd.DataFrame([{
            "Country": country,
            "Indicator": indicator,
            "Region": region,
            "adf_stat": adf_stat,
            "adf_p_value": p_value,
            "adf_stationary_flag": (
                "Stationary" if p_value < 0.05
                else "Non Stationary"
            )
        }])

    except Exception as e:
        return pd.DataFrame([{
            "Country": country,
            "Indicator": indicator,
            "Region": region,
            "adf_stat": None,
            "adf_p_value": None,
            "adf_stationary_flag": f"ADF Failed: {str(e)}"
        }])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

adf_schema = StructType([
    StructField("Country", StringType(), False),
    StructField("Indicator", StringType(), False),
    StructField("Region", StringType(), False),
    StructField("adf_stat", DoubleType(), True),
    StructField("adf_p_value", DoubleType(), True),
    StructField("adf_stationary_flag", StringType(), True)
])

adf_results = (compiled_drivers
                    .groupBy("Country","Indicator","Region")
                    .applyInPandas(adf_group, schema=adf_schema)
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### KPSS Test

# CELL ********************

def kpss_group(df):
    pdf = df.sort_values("Date")

    country = pdf["Country"].iloc[0]
    indicator = pdf["Indicator"].iloc[0]
    region = pdf["Region"].iloc[0]

    # FORCE CLEAN NUMERIC PIPELINE
    ts = (
        pd.to_numeric(pdf["Value"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .values
    )


    if len(ts) < 12:
        return pd.DataFrame([{
            "Country": country,
            "Indicator": indicator,
            "Region": region,
            "kpss_stat": None,
            "kpss_p_value": None,
            "kpss_stationary_flag": "Insufficient Data (<12 obs)"
        }])
    
    # SAFETY CHECK 2: constant series
    if np.nanstd(ts) == 0:
        return pd.DataFrame([{
            "Country": country,
            "Indicator": indicator,
            "Region": region,
            "kpss_stat": None,
            "kpss_p_value": None,
            "kpss_stationary_flag": "Constant Series (Skipped)"
        }])

        
    ## Run KPSS
    try:
        kpss_stat, p_value, _, _ = kpss(
            ts,
            regression="c",
            nlags="auto"
        )

        return pd.DataFrame([{
                "Country": country,
                "Indicator": indicator,
                "Region": region,
                "kpss_stat": kpss_stat,
                "kpss_p_value": p_value,
                "kpss_stationary_flag": 
                    "Stationary" if p_value >= 0.05
                    else "Non-Stationary"
        }])


    except Exception as e:
        return pd.DataFrame([{
                "Country": country,
                "Indicator": indicator,
                "Region": region,
                "kpss_stat": None,
                "kpss_p_value": None,
                "kpss_stationary_flag": f"KPSS Failed: {str(e)}"
        }])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

kpss_schema = StructType([
                    StructField("Country", StringType(), False),
                    StructField("Indicator", StringType(), False),
                    StructField("Region", StringType(), False),
                    StructField("kpss_stat", DoubleType(), True),
                    StructField("kpss_p_value", DoubleType(), True),
                    StructField("kpss_stationary_flag", StringType(), False)
])

kpss_results = (compiled_drivers.groupBy("Country", "Indicator","Region")
                    .applyInPandas(kpss_group, schema=kpss_schema))

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

df_stationary_stats = adf_results.join(kpss_results, ["Country", "Indicator","Region"])


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


## if both tests identified the driver as stationary then use directly for CCF otherwise apply differencing transformation
driver_final_stats = driver_final_stats.withColumn(
        "stationary_class",
        when((col("ADF_is_stationary")==1) & (col("KPSS_is_stationary")==1), "Stationary_use_levels")
        .when((col("ADF_is_stationary")==1) & (col("KPSS_is_stationary")==0), "Detrend")
        .otherwise("Log_diff")
    )\
    .drop("ADF_status","KPSS_status","ADF_is_stationary","KPSS_is_stationary")


driver_final_stats.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.driver_stats")

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

driver_classification = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.driver_stats")\
        .drop('adf_stat',
            'adf_p_value',
            'adf_stationary_flag',
            'kpss_stat',
            'kpss_p_value',
            'kpss_stationary_flag',)

feature_set_valid = broadcast(
    driver_classification.filter(col("valid_flag")==1)
).join(
    feature_set, ["Country","Indicator","Region"], "inner"
).drop("valid_flag","stationary_class")

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

topline_schema = StructType([
    StructField("Product_Category", StringType(), False),
    StructField("series", StringType(), False),
    StructField("Date", DateType(), False),
    StructField("Value", DoubleType(), True),
    StructField("residual", DoubleType(), True)
])

topline_residuals = topline_cutoff_data.groupBy("Product_Category","series")\
        .applyInPandas(
            stl_decompose,
            schema=topline_schema
        )

middle_schema = StructType([
    StructField("Product_Category", StringType(), False),
    StructField("Region", StringType(), False),
    StructField("series", StringType(), False),
    StructField("Date", DateType(), False),
    StructField("Value", DoubleType(), True),
    StructField("residual", DoubleType(), True)
])

middle_residuals = middle_cutoff_data.groupBy("Product_Category","Region","series")\
    .applyInPandas(stl_decompose, schema=middle_schema)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

feature_set_schema = StructType([
    StructField("Country", StringType(), False),
    StructField("Indicator", StringType(), False),
    StructField("Region", StringType(), False),
    StructField("Date", DateType(), False),
    StructField("Value", DoubleType(), True),
    StructField("residual", DoubleType(), True)
])

feature_set_residuals = feature_set_valid.groupBy("Country","Indicator", "Region")\
    .applyInPandas(stl_decompose, schema=feature_set_schema)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Stage 3: Lag/Lead Identification of Features
# #### Application of CCF
# 
# #### Joining target / drivers prior to CCF calculation

# CELL ********************

middle_residuals.columns

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

feature_set_residuals = feature_set_residuals\
    .drop("Value")\
    .withColumnsRenamed({"Date":"feature_date", "residual":"feature_residual","Region":"feature_region"})

topline_residuals = topline_residuals\
    .drop("Value")\
    .withColumnsRenamed({"Date":"target_date","residual":"target_residual","Region":"target_region"})
    
middle_residuals = middle_residuals\
    .drop("Value")\
    .withColumnsRenamed({"Date":"target_date","residual":"target_residual","Region":"target_region"})


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ##### Create mapping of features to targets

# CELL ********************

## Create valid mapping for target-driver combinations

## create distinct records of each df set
topline_distinct = topline_residuals.select("series").distinct()

middle_distinct = middle_residuals.select("series","target_region").distinct()

feature_world_distinct = feature_set_residuals.filter(col("feature_region")=="World").select("feature_region","Country","Indicator").distinct()
feature_m_distinct = feature_set_residuals.filter(~(col("feature_region")=="World"))\
    .select("feature_region","Country","Indicator").distinct()


## create pair dataframes
topline_pairs = broadcast(topline_distinct).crossJoin(feature_world_distinct)



middle_w_pairs = broadcast(middle_distinct).crossJoin(feature_world_distinct)
middle_pairs = broadcast(middle_distinct).join(
    feature_m_distinct,
    middle_distinct["target_region"]==feature_m_distinct["feature_region"],
    "inner"
)

middle_pairs = middle_pairs.unionByName(middle_w_pairs)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_pairs.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.topline_pairs")
middle_pairs.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.middle_pairs")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## Creating Expanded versions of the middle & topline residuals based on the pair mappings
    ## both approaches for managing schema/column explosion during join work
    
expanded_topline = topline_residuals.join(
    broadcast(topline_pairs),
    #topline_residuals["series"]==topline_pairs["series"],
    ["series"],
    "left"
)#.drop(topline_pairs["series"])

display(expanded_topline.limit(5))

expanded_middle = middle_residuals.join(
    broadcast(middle_pairs),
    (middle_residuals["series"]==middle_pairs["series"]) & 
    (middle_residuals["target_region"] == middle_pairs["target_region"]),
    "left"
).drop(middle_pairs["series"],middle_pairs["target_region"])

display(expanded_middle.limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

expanded_topline.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.expanded_topline")
expanded_middle.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.expanded_middle")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## Creating Expanded version of the middle & topline features based on the pair mappings
expanded_t_features = broadcast(topline_pairs).join(
    feature_set_residuals,
    ["feature_region","Indicator"],
    "inner"
).drop(feature_set_residuals["Country"])

display(expanded_t_features.limit(3))

expanded_m_features = broadcast(middle_pairs).join(
    feature_set_residuals,
    ["feature_region","Indicator","Country"],
    "inner"
)

display(expanded_m_features.limit(3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

expanded_t_features.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.expanded_t_features")
expanded_m_features.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.expanded_m_features")

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

## joining the feature and target residuals based on the pair mapping built within both dataframes
expanded_t_features = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.expanded_t_features")
expanded_m_features = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.expanded_m_features")


expanded_topline = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.expanded_topline")
expanded_middle = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.expanded_middle")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# expanded_topline = spark.createDataFrame(expanded_topline.rdd, expanded_topline.schema)
# expanded_middle = spark.createDataFrame(expanded_middle.rdd, expanded_middle.schema)

# expanded_t_features = spark.createDataFrame(expanded_t_features.rdd, expanded_t_features.schema)
# expanded_m_features = spark.createDataFrame(expanded_m_features.rdd, expanded_m_features.schema)


t = expanded_topline.alias("t")
m = expanded_middle.alias("m")
tf = expanded_t_features.alias("tf")
mf = expanded_m_features.alias("mf")


final_topline = t.join(
    tf,
    (t["series"] == tf["series"]) &
    (t["feature_region"] == tf["feature_region"]) &
    (t["Country"] == tf["Country"]) &
    (t["Indicator"] == tf["Indicator"]) &
    (t["target_date"] == tf["feature_date"]),
    "left"    
).select(t["series"],t["Product_Category"],t["target_date"],t["target_residual"],t["feature_region"],t["Country"],
        t["Indicator"],tf["feature_residual"])

middle_final = m.join(
    mf,
    (m["series"]==mf["series"]) &
    (m["target_region"] ==mf["target_region"]) &
    (m["feature_region"]==mf["feature_region"]) &
    (m["Country"]==mf["Country"]) &
    (m["Indicator"]==mf["Indicator"]) &
    (m["target_date"]==mf["feature_date"]),
    "left"
).select(m["series"],m["Product_Category"],m["target_region"],m["target_date"],m["target_residual"],
        mf["feature_region"],mf["Indicator"],mf["Country"],mf["feature_residual"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

final_topline.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.topline_w_features")
middle_final.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.middle_w_features")

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

# ### CCF Compute

# CELL ********************

def compute_ccf(pdf):

    pdf = pdf.sort_values("date")

    driver = pdf["driver"]
    target = pdf["target"]

    results = []


    for lag in range(-24, 25):

        if lag < 0:
            d = driver.iloc[:lag]
            t = target.iloc[-lag:]

        elif lag > 0:
            d = driver.iloc[lag:]
            t = target.iloc[:-lag]

        else:
            d = driver
            t = target

        corr = d.corr(t) if len(d) > 1 and len(t) > 1 else None

        if 'target_region' in pdf.columns:
            results.append({
                "series": pdf["series"].iloc[0],
                "Product_Category": pdf["Product_Category"].iloc[0],
                "target_region": pdf["target_region"].iloc[0],
                "feature_region": pdf["feature_region"].iloc[0],
                "Country": pdf["Country"].iloc[0],
                "Indicator": pdf["Indicator"].iloc[0],
                "Lag": lag,
                "Correlation": (
                                float(0.0)
                                if corr is None or pd.isna(corr) or np.isinf(corr)
                                else float(corr)
                            )
            })

        else:
            results.append({
                "series": pdf["series"].iloc[0],
                "feature_region": pdf["feature_region"].iloc[0],
                "Country": pdf["Country"].iloc[0],
                "Indicator": pdf["Indicator"].iloc[0],
                "Lag": lag,
                "Correlation": (
                                float(0.0)
                                if corr is None or pd.isna(corr) or np.isinf(corr)
                                else float(corr)
                            )
            })

    return pd.DataFrame(results)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_w_features = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.topline_w_features")\
    .withColumnsRenamed({"target_date":"date", "target_residual":"target", "feature_residual":"driver"})\
    .select("series","Product_Category","feature_region","Country","Indicator","date","target","driver")

middle_w_features = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.middle_w_features")\
    .withColumnsRenamed({"target_date":"date", "target_residual":"target", "feature_residual":"driver"})\
    .select("series","Product_Category","target_region","feature_region","Country","Indicator","date","target","driver")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_ccf_schema = (StructType([
    StructField("series", StringType(), False),
    StructField("feature_region", StringType(), False),
    StructField("Country", StringType(), False),
    StructField("Indicator", StringType(), False),
    StructField("Lag", IntegerType(), False),
    StructField("Correlation", DoubleType(), True)
]))

topline_ccf = topline_w_features.groupBy("series","feature_region","Country","Indicator")\
    .applyInPandas(compute_ccf, schema=topline_ccf_schema)


middle_ccf_schema = (StructType([
    StructField("series", StringType(), False),
    StructField("Product_Category", StringType(), True),
    StructField("target_region", StringType(), True),
    StructField("feature_region", StringType(), True),
    StructField("Country", StringType(), True),
    StructField("Indicator", StringType(), True),
    StructField("Lag", IntegerType(), True),
    StructField("Correlation", DoubleType(), True)
]))

middle_ccf = middle_w_features.groupBy("series","Product_Category","target_region","feature_region","Country","Indicator")\
    .applyInPandas(compute_ccf, schema=middle_ccf_schema)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_ccf.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.topline_ccf_base")
middle_ccf.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.middle_ccf_base")

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

topline_ccf = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.topline_ccf_base")
middle_ccf = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.middle_ccf_base")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print(topline_ccf.columns)
print(middle_ccf.columns)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(topline_ccf.select('series','feature_region','Country','Indicator').distinct().groupBy('series').count())
display(middle_ccf.select('series','feature_region','Country','Indicator').distinct().groupBy('series').count())


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def ccf_filtering(df):
    window = Window.partitionBy("series","feature_region","Country","Indicator")
    df = df.withColumn("abs_corr", abs(col("Correlation")))
    df = df.withColumn("max_corr", max(col("abs_corr")).over(window))


    ## filter out records with a correlations < .3 or Indicator is NaN
    df_filtered = df.filter(
        (col("max_corr") > 0.15) &
        # (col("max_corr") < .95) &
        (col("Indicator").isNotNull()) &
        (col("Indicator") != "NaN")
    )

    df_ranked = df_filtered.groupBy("series","feature_region","Country","Indicator").agg(first(col("max_corr")).alias("max_corr"))

    w = Window.partitionBy("series").orderBy(desc("max_corr"))

    df_ranked = df_ranked.withColumn("rank", rank().over(w))
    df_rank_filtered = df_ranked.filter(col("rank")<=500)

    df_final_filtered = df_rank_filtered.join(df_filtered, ['series','feature_region','Country','Indicator'], 'inner').drop(df_rank_filtered['max_corr'],df_rank_filtered['rank'])

    return df_final_filtered

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_ccf_filtered = ccf_filtering(topline_ccf)
middle_ccf_filtered = ccf_filtering(middle_ccf)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

topline_ccf_filtered.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.topline_ccf_filtered") 
middle_ccf_filtered.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.middle_ccf_filtered")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Correlation Clustering

# CELL ********************

def corr_clustering(df):

    serie = df['series'][0]
    pivot_cols = ["series","target_date"]
    middle_flag = False

    if 'target_region' in df.columns:
        pivot_cols.append("target_region")
        target_region = df['target_region'][0]
        middle_flag = True

    wide_df = (
        df.pivot_table(
            index=pivot_cols,
            columns="feature_id",
            values="feature_residual",
            aggfunc="first"
            ).reset_index()
        )

    
    


    X = wide_df.drop(columns=pivot_cols)
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median(numeric_only=True))

    corr = X.corr().abs()
    corr = corr.fillna(0)
    corr = np.clip(corr, 0, 1)

    ## convert correlation matrix to distance matrix
    distance = 1 - corr
    distance = np.clip(distance, 0, 1)



    ## hierarchical clustering 
    condensed_dist = squareform(distance.values, checks=False)

    linkage=sch.linkage(condensed_dist, method="average")

    threshold = 0.2
    cluster_labels = sch.fcluster(linkage,t=threshold, criterion="distance")
    
    if middle_flag:
        cluster_map = pd.DataFrame({
            "serie": serie,
            "target_region": target_region,
            "feature_id": X.columns,
            "cluster": cluster_labels
        })
    else:
        cluster_map = pd.DataFrame({
            "serie": serie,
            "feature_id": X.columns,
            "cluster": cluster_labels
        })

    representatives = []
    for cluster_id in cluster_map["cluster"].unique():
        features = cluster_map[cluster_map["cluster"] == cluster_id]["feature_id"].tolist()
        
        if len(features) == 1:
            representatives.append(features[0])
            continue

        sub_corr = corr.loc[features, features]

        # score = mean correlation to others (higher = more central)
        scores = sub_corr.mean(axis=1)

        rep = scores.idxmax()
        representatives.append(rep)

    X_reduced = X[representatives]

    final_df = pd.concat([df[["target_date"]], X_reduced], axis=1)


    return cluster_map
    


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def top_20_feature_extraction(df_f_resid, df_ccf_filtered):
    ## creating a key column for feature/serie combinations
    df_f_resid = df_f_resid.withColumn("feature_serie",
        concat_ws("__", "series","feature_region","Country","Indicator"))


    df_ccf_filtered = df_ccf_filtered.withColumn("feature_serie",
        concat_ws("__", "series","feature_region","Country","Indicator"))
    
    ## creating the feature_id for the feature/serie combinations that persisted after filtering
    feature_series = df_ccf_filtered.select("feature_serie").distinct()
    feature_series = feature_series.withColumn("feature_id", row_number().over(Window.orderBy("feature_serie")))

    ## ensuring there is no ambiguity in the joins
    r = df_f_resid.alias("r")


    initial_joined_df = feature_series.join(df_ccf_filtered, ["feature_serie"], "left")
    joined_df = initial_joined_df.join(df_f_resid, ['feature_serie'], 'inner')\
        .drop(r["series"],r["feature_region"],r["Country"],r["Indicator"],r["feature_serie"])



    ## updating logic to create the wide df later within the corr_clustering function
    middle_process = False
    if 'target_region' in joined_df.columns:
        middle_process = True


    if middle_process:
        print("middle process")
        middle_schema = StructType([
            StructField("serie", StringType(), False),
            StructField("target_region", StringType(), False),
            StructField("feature_id", IntegerType(), False),
            StructField("cluster", IntegerType(), False)
        ])
        clustering = joined_df.groupBy("series","target_region").applyInPandas(corr_clustering, schema=middle_schema)

    else:
        print("topline_process")
        topline_schema = StructType([
            StructField("serie", StringType(), False),
            StructField("feature_id", IntegerType(), False),
            StructField("cluster", IntegerType(), False)
        ])

        clustering = joined_df.groupBy("series").applyInPandas(corr_clustering, schema=topline_schema)



    ij = initial_joined_df.alias("ij")
    c = clustering.alias("c")

    if middle_process:
        data_w_cluster = ij.join(
            c,
            (ij['feature_id']==c['feature_id']) & (ij['series']==c['serie']) & (ij['target_region']==c['target_region']),
            'inner'
        ).drop(c['feature_id'],c['serie'],c['target_region'])

    else:
        data_w_cluster = ij.join(
            c,
            (ij['feature_id']==c['feature_id']) & (ij['series']==c['serie']),
            'inner'
        ).drop(c['feature_id'],c['serie'])

    

    cluster_corr_w = Window.partitionBy("series", "cluster").orderBy(desc("max_corr"))

    data_distinct = data_w_cluster.select("series","feature_id","cluster","max_corr").distinct()

    data_distinct = data_distinct.withColumn("cluster_rank", row_number().over(cluster_corr_w))

    data_filtered = data_distinct.filter(col("cluster_rank")<=3)


    ## joining remaining features w/ original data
    df = data_filtered.alias('df')

    post_cluster_filtering = df.join(ij, ['feature_id'], 'inner').drop(df['series'],df['max_corr'])

    ## ranking series within clusters
    ranked_w = Window.partitionBy("series").orderBy(desc('max_corr'))

    post_cluster_filtering = post_cluster_filtering.withColumn("feature_rank", dense_rank().over(ranked_w))

    top_20_features = post_cluster_filtering.filter(col("feature_rank")<=20).select("series","feature_region","Country","Indicator","Lag","Correlation","abs_corr","max_corr","feature_rank")
    top_20_features = top_20_features.withColumn("rec_lag", 
        when(col("abs_corr")==col("max_corr"),lit(1)).otherwise(lit(0)))


    return top_20_features



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

middle_w_features = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.middle_w_features")\
    .select("series","feature_region","Country","Indicator","target_date","feature_residual")
topline_w_features = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.topline_w_features")\
    .select("series","feature_region","Country","Indicator","target_date","feature_residual")

middle_ccf_filtered = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.middle_ccf_filtered")
topline_ccf_filtered = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.topline_ccf_filtered")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

final_middle_features = top_20_feature_extraction(middle_w_features, middle_ccf_filtered)

final_topline_features = top_20_feature_extraction(topline_w_features, topline_ccf_filtered)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

final_topline_features.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.topline_20_features")
final_middle_features.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.Driver_Exploration_V2.middle_20_features")

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

# ### 5. Complete Economic Validation
# - validate choices with Dominik for feature selection
# - Save the target time series / feature selection as output

# CELL ********************

topline_feature_selection = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.topline_20_features")
topline_feature_selection = topline_feature_selection.withColumn("feature_serie", concat_ws("__","feature_region","Country","Indicator"))
print(topline_feature_selection.columns)
display(topline_feature_selection.select('series').distinct())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(topline_feature_selection.filter(col("series")=="ALU"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

middle_feature_selection = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.middle_20_features")
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
