# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {}
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

# ### Topline investigation

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

## filter based on which serie you want to look into based on the distinct serie output above
display(topline_feature_selection.filter(col("series")=="SCREWS"))

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
