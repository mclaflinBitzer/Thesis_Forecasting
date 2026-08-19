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

import pandas as pd
from pyspark.sql.functions import *

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

base_folder = "abfss://991f5e4b-c174-4ff2-992e-feb17d49d25a@onelake.dfs.fabric.microsoft.com/22746de3-183e-4327-a844-dceda0b7165c/Files/Automated_Driver_Analysis"

classic_features = spark.createDataFrame(
    pd.read_excel(base_folder + "/Middle/middle_ENCV_selected_features.xlsx")
)

ml_features = spark.createDataFrame(
    pd.read_excel(base_folder + "/Middle/middle_xgboost_selected_features.xlsx")
)

dl_features = spark.createDataFrame(
    pd.read_excel(base_folder + "/Topline/topline_dl_selected_feature.xlsx")
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(dl_features.groupBy('Product_Category').agg(countDistinct('Indicator')))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(classic_features.groupBy('Product_Category','Region').agg(countDistinct('Feature')))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

dl_new = dl_features.filter(col('rank')<=30)
display(dl_new.groupBy('Product_Category','Region').agg(countDistinct('Indicator')))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(dl_features.limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(dl_features.groupBy('Product_Category','Region').agg(countDistinct('Indicator')))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
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

# CELL ********************

driver_test = spark.read.table("Sales_Forecasting.silver.compiled_drivers").filter(
    (col("Indicator")=="Data Center") |
    (col("Indicator")=="Cold Storage Plants") |
    (col("Indicator")=="Food Processing Plants") |
    (col("Indicator")=="Leisure & Hospitality Buildings") |
    (col("Indicator")=="Stores") 
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print(driver_test.columns)

pivot = (
    driver_test
    .groupBy("Date")
    .pivot("Indicator")
    .agg(first("Value"))
    .orderBy("Date")
)

pivot_flagged = pivot.withColumn(
    'same_value_flag', 
    when(
        (col("Data Center")==col("Food Processing Plants")) |
        (col("Food Processing Plants")==col("Leisure & Hospitality Buildings")) |
        (col("Leisure & Hospitality Buildings")==col("Stores")),
         lit(1)
        ).otherwise(lit(0))
)

display(pivot_flagged.filter(col("same_value_flag")==1).limit(50))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Topline investigation

# CELL ********************

topline_feature_selection = spark.read.table("Sales_Forecasting.Driver_Exploration_V2.topline_top_features")
display(topline_feature_selection.select('series').distinct())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## filter based on which serie you want to look into based on the distinct serie output above
display(topline_feature_selection.filter(col("series")=="ALU").filter(
    (col("Indicator")=="Data Center") |
    (col("Indicator")=="Cold Storage Plants") |
    (col("Indicator")=="Food Processing Plants") |
    (col("Indicator")=="Leisure & Hospitality Buildings") |
    (col("Indicator")=="Stores") 
))

## from the output click on "New Chart"
## create line chart, with Lag on the X axis and Correlation on the Y axis
## add feature_serie to the "serie" label and remove the legend from the visualization

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Middle investigation

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

## filter based on which serie you want to look into based on the distinct serie output above
display(middle_feature_selection.filter(col("series")=='PISTON___APAC'))

## from the output click on "New Chart"
## create line chart, with Lag on the X axis and Correlation on the Y axis
## add feature_serie to the "serie" label and remove the legend from the visualization

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Horvath Investigation

# CELL ********************

path = "/lakehouse/default/Files/Driver_Analysis/Horvath_topline_feature_analysis.xlsx"

horvath_features = spark.createDataFrame(pd.read_excel(path))
display(horvath_features)

## from the output click on "New Chart"
## create line chart, with Lag on the X axis and Correlation on the Y axis
## add Indicator to the "serie" label and remove the legend from the visualization

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import os


print(os.path.exists(path))

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
