# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {}
# META }

# CELL ********************

RAW_TABLE_NAME = "Sales_Forecasting.bronze.Raw_Analyse_Sales_BPC"
FILTERED_TABLE_NAME = "Sales_Forecasting.bronze.Filtered_Data"
REGION_MAPPING_TABLE_NAME ="`BSI - Marketing`.Sales_Report.dbo.dim_sales_office"

model_data_mapping = {
    "filtered":{
        "grouping_cols": ["Product_Category", "Housing_Size", "Region"],
        "table_name": "Sales_Forecasting.bronze.Filtered_Data",
        "target_col": "Quantity"
    },
    "topline": {
        "grouping_cols": ["Product_Category"],
        "table_name": "Sales_Forecasting.bronze.Topline_Data",
        "target_col": "Quantity",
    },
    "middle":{
        "grouping_cols": ["Product_Category", "Region"],
        "table_name": "Sales_Forecasting.bronze.Middle_Data",
        "target_col": "Quantity",
    }
} 


CUSUM_TABLE = "Sales_Forecasting.Data_Exploration.structual_break_cusum"
PELT_TABLE = "Sales_Forecasting.Data_Exploration.pelt_break"
ROLLING_STATS_TABLE = "Sales_Forecasting.Data_Exploration.rolling_stats"
RAW_OBS = "Sales_Forecasting.Data_Exploration.base_observations"
STABLE_STATS_TABLE = "Sales_Forecasting.Data_Exploration.stable_run_stats"
STABLE_RUNS_TABLE = "Sales_Forecasting.Data_Exploration.stable_runs"
BASE_DQ_TABLE = "Sales_Forecasting.Data_Exploration.base_data_quality"

TOPLINE_CUTOFF_DATA_TABLE = "Sales_Forecasting.silver.topline_cutoff_data" 
MIDDLE_CUTOFF_DATA_TABLE = "Sales_Forecasting.silver.middle_cutoff_data"

CUTOFF_DQ_TABLE = "Sales_Forecasting.Data_Exploration.cutoff_data_quality"
STL_METRICS_TABLE = "Sales_Forecasting.Data_Exploration.STL_metrics"
ADF_STATS_TABLE = "Sales_Forecasting.Data_Exploration.ADF_stats"
KPSS_STATS_TABLE = "Sales_Forecasting.Data_Exploration.kpss_stats"
PP_STATS_TABLE = "Sales_Forecasting.Data_Exploration.pp_stats"
STATIONARY_STATS_TABLE = "Sales_Forecasting.Data_Exploration.stationary_stats"

HISTORICAL_CUTOFF_DATES = {
    "ALU": "01-03-2019",
    "AVP_CDU": "01-05-2016",
    "HEXPV": "01-01-2018",
    "MAERSK_COMPRESSOR": "01-02-2020",
    "MAERSK_ELECTRONICS": "01-10-2019",
    "PISTON": "01-10-2016",
    "SCREWS": "01-04-2020",
    "SCROLLS": "01-06-2015"
}

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
