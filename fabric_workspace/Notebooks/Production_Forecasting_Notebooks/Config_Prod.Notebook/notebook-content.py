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

# ### HISTORICAL CUTOFF DATES

# CELL ********************

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

HIST_CUTOFF_DATE_TABLE = "Sales_Forecasting.silver.historical_cutoff_dates"


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Static Configuration

# CELL ********************

## Manual Driver Feature Engineering / Selection Config
manual_driver_parquet_base_dir = 'abfss://991f5e4b-c174-4ff2-992e-feb17d49d25a@onelake.dfs.fabric.microsoft.com/22746de3-183e-4327-a844-dceda0b7165c/Files/Driver_Analysis/'
manual_driver_excel_base_dir = "/lakehouse/default/Files/Driver_Analysis/"
manual_driver_feature_num = 30
driver_table = "Sales_Forecasting.silver.compiled_drivers"

## Automated Driver Feature Engineering / Selection config
## Declaring which feature selection processes run 
    # ElasticNetCV -> Classical Statistical Models
    # XGBoost -> ML Models
    # DL -> DL Models
elasticnet_run = True
xgboost_run = True
dl_run = True
automated_drv_excel_base_dir = "/lakehouse/default/Files/Automated_Driver_Analysis/"
elasticnet_init_features = 10
xgboost_init_features = 25
dl_init_features = 30

## Forecasting config
manual_features_output_base_dir = "/lakehouse/default/Files/Driver_Analysis/Final_Feature_Selection/"
automated_features_output_base_dir= "/lakehouse/default/Files/Automated_Driver_Analysis/"
forecast_parquet_base_dir = "abfss://991f5e4b-c174-4ff2-992e-feb17d49d25a@onelake.dfs.fabric.microsoft.com/22746de3-183e-4327-a844-dceda0b7165c/Files/Forecasting"



VALID_DRIVER_STATUS = [
    "No_Drivers",
    "Manual_Drivers",
    "Automated_Drivers"
]

FORECAST_HORIZON = 18
SEASONAL_PERIODS = 12
MIN_TRAIN = 36
Classical_STEP_SIZE = 1
ML_STEP_SIZE = 3
N_OPTUNA_TRIALS = 50
OPTUNA_SEED = 42
ENCODER_LENGTH = 36

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Forecasting Level Configuration

# CELL ********************



model_data_mapping = {
    "topline": {
        "grouping_cols": ["Product_Category"],
        "init_act_table": "Sales_Forecasting.bronze.Topline_Data",
        "target_col": "Quantity",
        "raw_act_table": "Sales_Forecasting.bronze.topline_raw",
        'cutoff_data_table': "Sales_Forecasting.silver.topline_cutoff_data" ,
        'manual_features_output_dir': manual_features_output_base_dir + "final_features_topline.csv",
        'classic_automated_features_output_dir': automated_features_output_base_dir + "Topline/topline_ENCV_selected_features.xlsx",
        'xgboost_automated_features_output_dir': automated_drv_excel_base_dir + "Topline/topline_xgboost_selected_feature.xlsx",
        'forecast_parquet_base_dir': forecast_parquet_base_dir + "/Topline/",

    },
    "middle":{
        "grouping_cols": ["Product_Category", "Region"],
        "init_act_table": "Sales_Forecasting.bronze.Middle_Data",
        "target_col": "Quantity",
        'raw_act_table': 'Sales_Forecasting.bronze.middle_raw',
        'cutoff_data_table': "Sales_Forecasting.silver.middle_cutoff_data",
        'manual_features_output_dir': manual_features_output_base_dir + "final_features_middle.csv",
        'classic_automated_features_output_dir': automated_features_output_base_dir + "Middle/middle_ENCV_selected_features.xlsx",
        'xgboost_automated_features_output_dir': automated_drv_excel_base_dir + "Middle/middle_xgboost_selected_feature.xlsx",
        'forecast_parquet_base_dir': forecast_parquet_base_dir + "/Middle/",
    }
} 


manual_driver_mapping = {
    'topline': {
        'drv_grp_cols': ['Indicator'],
        'act_grp_cols': ['Product_Category','series'],
        'drv_rename': {"Date":"feature_date","residual":"feature_residual"},
        'act_rename': {"Date":"target_date","residual":"target_residual"},
        'stationary_stats_output_path' : manual_driver_excel_base_dir + "topline_stationary_stats.xlsx",
        'pairs_path' : manual_driver_excel_base_dir + "topline_pairs.xlsx",
        'joined_act_f_dir' : manual_driver_parquet_base_dir + "final_topline.parquet",
        'ccf_output_dir' : manual_driver_parquet_base_dir + 'topline_ccf_base.parquet',
        'final_feature_dir' : manual_driver_excel_base_dir + 'topline_recommended_features.xlsx',
        'recommended_features_path': '/lakehouse/default/Files/Driver_Analysis/topline_recommended_features.xlsx',
        'final_features_path': "/lakehouse/default/Files/Driver_Analysis/Final_Feature_Selection/final_features_topline.csv",
    
    },
    'middle':{
        'drv_grp_cols': ['Region','Indicator'],
        'act_grp_cols': ['Product_Category', 'Region','series'],
        'drv_rename': {"Date":"feature_date","residual":"feature_residual"},
        'act_rename': {"Date":"target_date","residual":"target_residual"},
        'stationary_stats_output_path' : manual_driver_excel_base_dir + "middle_stationary_stats.xlsx",
        'pairs_path' : manual_driver_excel_base_dir + "middle_pairs.xlsx",
        'joined_act_f_dir' : manual_driver_parquet_base_dir + "final_middle.parquet",
        'ccf_output_dir' : manual_driver_parquet_base_dir + 'middle_ccf_base.parquet',
        'final_feature_dir' : manual_driver_excel_base_dir + 'middle_recommended_features.xlsx',
        'recommended_features_path': '/lakehouse/default/Files/Driver_Analysis/middle_recommended_features.xlsx',
        'final_features_path': "/lakehouse/default/Files/Driver_Analysis/Final_Feature_Selection/final_features_middle.csv",
    }
}

automated_driver_mapping = {
    'topline': {
        'act_cols_rename': {"Date":"target_date","residual":"target_residual"}, 
        'drv_cols_rename': {"Date":"feature_date","residual":"feature_residual"},
        'feature_diagnostics_file': automated_drv_excel_base_dir + "Topline/topline_ENCV_feature_diagnostics.xlsx",
        'selected_feature_file' : automated_drv_excel_base_dir + "Topline/topline_ENCV_selected_features.xlsx",
        'xgboost_diagnostics_file' : automated_drv_excel_base_dir + "Topline/topline_xgboost_feature_diagnostics.xlsx",
        'xgboost_selected_feature_file' : automated_drv_excel_base_dir + "Topline/topline_xgboost_selected_feature.xlsx",
        'dl_selected_feature_file' : automated_drv_excel_base_dir + "Topline/topline_dl_selected_feature.xlsx",

    },
    'middle': {
        'act_cols_rename': {"Date":"target_date","residual":"target_residual"}, 
        'drv_cols_rename': {"Date":"feature_date","residual":"feature_residual"},
        'feature_diagnostics_file': automated_drv_excel_base_dir + "Middle/middle_ENCV_feature_diagnostics.xlsx",
        'selected_feature_file' : automated_drv_excel_base_dir + "Middle/middle_ENCV_selected_features.xlsx",
        'xgboost_diagnostics_file' : automated_drv_excel_base_dir + "Middle/middle_xgboost_feature_diagnostics.xlsx",
        'xgboost_selected_feature_file' : automated_drv_excel_base_dir + "Middle/middle_xgboost_selected_feature.xlsx",
        'dl_selected_feature_file' : automated_drv_excel_base_dir + "Middle/middle_dl_selected_feature.xlsx",
    }
}


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Building Runtime Configuration

# CELL ********************


def build_config(
    topline: bool,
    driver_status: str
):

    if driver_status not in VALID_DRIVER_STATUS:
        raise ValueError(
            f"Invalid driver_status={driver_status}. "
            f"Expected one of {VALID_DRIVER_STATUS}"
        )

    level = "topline" if topline else "middle"

    ## forecasting parameter selection
    level_config = model_data_mapping[level]
    init_act_table = level_config['init_act_table']
    target_col = level_config['target_col']
    grouping_cols = level_config['grouping_cols']
    raw_act_table = level_config['raw_act_table']
    cutoff_data_table = level_config['cutoff_data_table']
    manual_features_output_dir = level_config['manual_features_output_dir']
    classic_automated_features_output_dir = level_config['classic_automated_features_output_dir']
    xgboost_automated_features_output_dir = level_config['xgboost_automated_features_output_dir']
    forecast_parquet_base_dir = level_config['forecast_parquet_base_dir']

    ## Manual FE / FS parameters
    level_manual_fe = manual_driver_mapping[level]
    manual_drv_grp_cols = level_manual_fe['drv_grp_cols'] 
    manual_act_grp_cols = level_manual_fe['act_grp_cols']
    manual_act_rename = level_manual_fe['act_rename']
    manual_drv_rename = level_manual_fe['drv_rename']

    manual_stationary_stats_output_path = level_manual_fe['stationary_stats_output_path']
    manual_pairs_path = level_manual_fe['pairs_path']
    manual_joined_act_f_dir = level_manual_fe['joined_act_f_dir']
    manual_ccf_output_dir = level_manual_fe['ccf_output_dir']
    manual_final_feature_dir = level_manual_fe['final_feature_dir']
    manual_recommended_features_path = level_manual_fe['recommended_features_path']
    manual_final_features_path = level_manual_fe['final_features_path']

    ## Automated FE / FS parameters
    level_automated_fe = automated_driver_mapping[level]
    automated_act_cols_rename = level_automated_fe['act_cols_rename']
    automated_drv_cols_rename = level_automated_fe['drv_cols_rename']
    automated_feature_diagnostics_file = level_automated_fe['feature_diagnostics_file']
    automated_selected_feature_file = level_automated_fe['selected_feature_file']
    automated_xgboost_diagnostics_file = level_automated_fe['xgboost_diagnostics_file']
    automated_xgboost_selected_feature_file = level_automated_fe['xgboost_selected_feature_file']
    automated_dl_selected_feature_file = level_automated_fe['dl_selected_feature_file']    



    # # Select driver feature file
    # if driver_status == "Manual_Drivers":
    #     selected_driver_dir = level_config["manual_features"]

    # elif driver_status == "Automated_Drivers":
    #     selected_driver_dir = level_config["automated_features"]

    # else:
    #     selected_driver_dir = None

    # # Output directory
    # parquet_dir = (
    #     f"{BASE_PARQUET_DIR}/"
    #     f"{level.capitalize()}/"
    #     f"{driver_status}/"
    # )

    return {
        # Runtime selection
        "level": level,
        "driver_status": driver_status,
        "init_act_table": init_act_table,
        'target_col': target_col,
        'grouping_cols': grouping_cols,
        'raw_act_table': raw_act_table,
        'cutoff_data_table': cutoff_data_table,
        'manual_features_output_dir': manual_features_output_dir,
        'classic_automated_features_output_dir': classic_automated_features_output_dir,
        'xgboost_automated_features_output_dir': xgboost_automated_features_output_dir,
        'forecast_parquet_base_dir': forecast_parquet_base_dir,


        ## Manual FE/FS selection
            'manual_drv_grp_cols': manual_drv_grp_cols,
            'manual_act_grp_cols' : manual_act_grp_cols,
            'manual_act_rename' : manual_act_rename,
            'manual_drv_rename' : manual_drv_rename,

            'manual_stationary_stats_output_path' : manual_stationary_stats_output_path,
            'manual_pairs_path' : manual_pairs_path,
            'manual_joined_act_f_dir' : manual_joined_act_f_dir,
            'manual_ccf_output_dir' : manual_ccf_output_dir,
            'manual_final_feature_dir' : manual_final_feature_dir,
            'manual_recommended_features_path': manual_recommended_features_path,
            'manual_final_features_path': manual_final_features_path,


        ## Automated FE / FS selection
        'automated_act_cols_rename' : automated_act_cols_rename,
        'automated_drv_cols_rename' : automated_drv_cols_rename,
        'automated_feature_diagnostics_file' : automated_feature_diagnostics_file,
        'automated_selected_feature_file' : automated_selected_feature_file,
        'automated_xgboost_diagnostics_file' : automated_xgboost_diagnostics_file,
        'automated_xgboost_selected_feature_file' : automated_xgboost_selected_feature_file,
        'automated_dl_selected_feature_file' :  automated_dl_selected_feature_file ,


        # # Data configuration
        # "actuals_table": level_config["table_name"],
        # "initial_target_col": level_config["target_col"],
        # "driver_grouping_cols": level_config["driver_grouping_cols"],
        # "actual_grouping_cols": level_config["actual_grouping_cols"],
        # "selected_driver_dir": selected_driver_dir,

        # # Drivers
        # # "driver_table": DRIVER_TABLE,
        # "drivers_used": driver_status != "No_Drivers",

        # # Outputs
        # "parquet_dir": parquet_dir,

        # "TFT_forecast_dir": (
        #     parquet_dir + "TFT_Forecast_Output.parquet"
        # ),
        # "TFT_static_dir": (
        #     parquet_dir + "TFT_static_Output.parquet"
        # ),
        # "TFT_decoder_dir": (
        #     parquet_dir + "TFT_decoder_Output.parquet"
        # ),
        # "TFT_attention_dir": (
        #     parquet_dir + "TFT_attention_Output.parquet"
        # ),

        # # Model parameters
        # "forecast_horizon": FORECAST_HORIZON,
        # "seasonal_periods": SEASONAL_PERIODS,
        # "min_train": MIN_TRAIN,
        # "tune_holdout_months": TUNE_HOLDOUT_MONTHS,
        # "n_optuna_trials": N_OPTUNA_TRIALS,
        # "optuna_seed": OPTUNA_SEED,
        # "encoder_length": ENCODER_LENGTH,
        # "target_col": TARGET_COL,
    }

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
