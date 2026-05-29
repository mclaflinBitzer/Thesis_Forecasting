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
