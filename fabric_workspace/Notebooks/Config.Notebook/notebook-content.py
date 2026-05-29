# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {}
# META }

# MARKDOWN ********************

# # Data Loading / Preprocessing Configurations

# MARKDOWN ********************

# #### Raw Data - Configurations

# CELL ********************

_FABRIC_COLUMN_RENAME = {
    "CATEGORY Stufe 01.Schlüssel": "Category",
    "YEAR Stufe 01": "Year",
    "YEAR Stufe 01.Schlüssel": "Year_Stufe",
    "BASE_PERIOD Stufe 01": "Month",
    "SOLD_TO_PARTY Stufe 01.Schlüssel": "Sold_To_Party_Id",
    "SOLD_TO_PARTY Stufe 01": "Sold_To_Party",
    "PARTY_COUNTRY Stufe 01": "Sold_To_Party_Country",
    "PARTY_COUNTRY Stufe 01.Schlüssel": "Sold_To_Party_Country_Id",
    "CURRENCY Stufe 01.Schlüssel": "Currency",
    "PRODUCT_CATEGORY Stufe 01.Schlüssel": "Product_Category_Id",
    "PRODUCT_CATEGORY Stufe 01": "Product_Category",
    "SALES_OFFICE Stufe 01.Schlüssel": "Sales_Office_Id",
    "SALES_OFFICE Stufe 01": "Sales_Office",
    "ACCOUNT Stufe 01.Schlüssel": "Account_Id",
    "ACCOUNT Stufe 01": "Account_Name",
    "HOUSING_SIZE Stufe 01.Schlüssel": "Housing_Size_Id",
    "HOUSING_SIZE Stufe 01": "Housing_Size",
    "Menge": "Quantity",
    "Umsatz": "Value",
    'CATEGORY Stufe 01': "CATEGORY_Stufe_01",
    'CURRENCY Stufe 01': "CURRENCY_Stufe_01",
    'PLANNING_PRODUCTS_HS Stufe 01': 'PLANNING_PRODUCTS_HS_Stufe_01',
    'PLANNINGPRODUCTS_Stufe_01.Schlüssel': 'PLANNING_PRODUCTS_Stufe_01',
    'PLANNING_PRODUCTS_HS_Stufe 01.Schlüssel': 'PLANNING_PRODUCTS_HS_Stufe_01',
    'PLANNINGPRODUCTS Stufe 01.Schlüssel': 'PLANNING_PRODUCTS_Stufe_01_Id',
    'PLANNING_PRODUCTS_HS Stufe 01.Schlüssel': 'PLANNING_PRODUCTS_HS_Stufe_01_Id',
    'PLANNINGPRODUCTS Stufe 01': 'PLANNING_PRODUCTS_Stufe_01'
}

# Fabric-specific filter columns (dropped after filtering)
_FABRIC_FILTER_COLS = [
    "CATEGORY_Stufe_01",
    "CURRENCY_Stufe_01",
    "Category",
    "Currency",
    'PLANNING_PRODUCTS_HS_Stufe_01_Id',
    'PLANNING_PRODUCTS_Stufe_01_Id',
    "PLANNING_PRODUCTS_Stufe_01",
    "PLANNING_PRODUCTS_HS_Stufe_01", 
    "Year_Stufe"]

# The product categories to select, filter out all others
PRODUCT_CATEGORIES = [
    "ALU",
    "HEXPV",
    "AVP_CDU",
    "SCROLLS",
    "SCREWS",
    "PISTON",
    "MAERSK_COMPRESSOR",
    "MAERSK_ELECTRONICS",
]

RAW_TABLE_NAME = "Sales_Forecasting.bronze.Raw_Analyse_Sales_BPC"
FILTERED_TABLE_NAME = "Sales_Forecasting.bronze.Filtered_Data"
REGION_MAPPING_TABLE_NAME ="`BSI - Marketing`.Sales_Report.dbo.dim_sales_office"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Actual processing for model level aggregation

# CELL ********************

model_data_mapping = {
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
