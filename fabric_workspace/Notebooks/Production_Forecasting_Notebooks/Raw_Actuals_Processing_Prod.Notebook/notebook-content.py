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
# META         },
# META         {
# META           "id": "0eb8b523-56c5-454e-a585-6ae1a651318e"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

from pyspark.sql.functions import *
import pandas as pd
from pyspark.sql.types import *
from pyspark.sql.utils import AnalysisException
import numpy as np

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# PARAMETERS CELL ********************

## Parameters to set from the pipeline
Topline = True
driver_status = "Manual_Drivers"    ## options: "No_Drivers", "Manual_Drivers", "Automated_Drivers" 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# this run the "Config" file where pipeline variables are set, making them available for reference and usage
# 
# the imported variables include
# 
# **Part 1**
# - _FABRIC_COLUMN_RENAME
# - _FABRIC_FILTER_COLS
# - PRODUCT_CATEGORIES
# - RAW_TABLE_NAME
# - FILTERED_TABLE_NAME
# - REGION_MAPPING_TABLE_NAME
# 
# **Part 2**
# - model_data_mapping (contains the grouping columns and table name for the topline and middle models raw data)

# CELL ********************

%run Config_Prod

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

config_obj = build_config(
    topline=Topline,
    driver_status=driver_status
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

GRP_COLS = config_obj['grouping_cols']
TARGET_COL = config_obj['target_col']
init_act_table = config_obj['init_act_table']
raw_act_table = config_obj['raw_act_table']
cutoff_data_table = config_obj['cutoff_data_table']

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Step 1 Methods

# MARKDOWN ********************

# #### load_actuals_from_lakehouse method

# CELL ********************

def load_actuals_from_lakehouse(table_name, col_mapping, filter_cols):
    
    """Load actuals from a Lakehouse Delta table and normalize to pipeline format.

    Reads the raw BPC actuals Delta table, renames Fabric column names to the
    names expected by the pipeline, applies Fabric-specific account filtering
    (Z0002 and Z0004 split logic), creates the Date column, and returns a
    DataFrame ready to be passed directly to preprocess_actuals_data().

    Parameters
    ----------
    table_name : str
        Name of the Delta table in the attached Lakehouse. Default: "actuals".
    col_mapping: list
        mapping of what the original column names are renamed to for the Fabric environment 
    Returns
    -------
    pd.DataFrame
        Columns include: Category, Year, Month, Date, Sold_to_party_Id,
        Sold_to_party_Name, Sold_to_party_Country, Currency,
        Product_Category_Id, Product_Category, Sales_Office_Id,
        Sales_Office_Name, Account_Id, Account_Name, Housing_size_Id,
        Housing_size_name, Quantity, Value.
    """

    df = spark.read.table(table_name)

    # rename Fabric column names
    for old_col, new_col in col_mapping.items():
        df = df.withColumnRenamed(old_col, new_col)

    # validate if columns are missing
    required_after_rename = [
        "Year",
        "Month",
        "Account_Id",
        "PLANNING_PRODUCTS_Stufe_01",
        "PLANNING_PRODUCTS_HS_Stufe_01",
    ]

    missing_cols = [c for c in required_after_rename if c not in df.columns]
    
    if missing_cols:
        raise KeyError(f"Missing required columns after rename in load_actuals_from_lakehouse: {missing_cols}")

    ## Create Date column from Year (int) and Month (int)
    df = df.withColumn("Year", col("Year").cast("int"))
    df = df.withColumn("Month", col("Month").cast("int"))

    ## drop records where Year or Month are NA
    df = df.dropna(subset=["Year", "Month"])

    ## Generate a "Date" column based on Year and Month & populate as first of month
    df = df.withColumn("Date", make_date(col("Year"), col("Month"),lit(1)))
    
    ## Fabric account filtering
    ## Z0002: keep only rows where PLANNING_PRODUCTS == "Y" and PLANNING_PRODUCTS_HS =="#"
    ## This is equivelant to BPC's actuals only view

    result_df = df.filter(
            (
                (col("Account_Id")=="Z0002") &
                (col("PLANNING_PRODUCTS_Stufe_01") == "Y") &
                (col("PLANNING_PRODUCTS_HS_Stufe_01") == "#")
            ) |
            (
                (col("Account_Id") == "Z0004") &
                (col("PLANNING_PRODUCTS_HS_Stufe_01") == "#")
            )
        )

    ## drop unneeded columns
    result_df = result_df.drop(*[c for c in filter_cols if c in result_df.columns])
    
    return result_df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### preprocess_actuals_data method

# CELL ********************

def preprocess_actuals_data(df, product_categories=PRODUCT_CATEGORIES):
    print("Preprocessing Sales Actuals")
    print(f"Actual intital length: {df.count()}")
    
    # filters to only use "actual" real data points
    df = df.filter(col("Category")=="Actual")
    print(f"After filtering for only 'actual' / real records: {df.count()}")

    ## filtering intercompany transactions
    df = df.filter(
        # keep rows WITH letters
        col("Sold_to_party_Id").cast("string").rlike("[A-Za-z]")
        |
        # OR keep rows WITHOUT letters AND in numeric range
        (
            ~col("Sold_to_party_Id").cast("string").rlike("[A-Za-z]")
            &
            (
                (col("Sold_to_party_Id").cast("int") <= 300000)
                |
                (col("Sold_to_party_Id").cast("int") > 400000)
            )
        )
    ).orderBy("Sold_to_party_Id")
    print(f"AFter filtering intercompany transactions: {df.count()}")

    ## filter out LC currency 
    df = df.filter(col("Currency") != "LC")
    print(f"After filtering LC currency: {df.count()}")

    ## filtering for only the product categories that we want to forecast on
    if PRODUCT_CATEGORIES:
        df = df.filter(col("Product_Category").isin(PRODUCT_CATEGORIES))
        print(f"After keeping configured product categories: {df.count()}")
    else:
        print("No configured product categories provided, skipping category filtering")

    ## removing SDN1000 sales office
    df = df.filter(col("Sales_Office_Id") != "SDN1000")
    print(f"After filtering SDN1000: {df.count()}")

    return df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Write to Lakehouse Table

# CELL ********************

def write_to_lakehouse(df, table_name):
    """
    Writes df to a Lakehouse table.
    - If table exists: overwrite it
    - If table does not exist: create it
    """

    table_exists = False

    # --- Check if table exists ---
    try:
        spark.table(table_name)
        table_exists = True
    except AnalysisException:
        table_exists = False

    # --- Write logic ---
    if table_exists:
        print(f"Table '{table_name}' exists. Overwriting...")
        (
            df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(table_name)
        )
    else:
        print(f"Table '{table_name}' does not exist. Creating new table...")
        (
            df.write
            .format("delta")
            .mode("overwrite")   # overwrite is fine for first write
            .option("overwriteSchema", "true")
            .saveAsTable(table_name)
        )

    print("Write complete.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Merging w/ Region mapping from Sales Report dim table

# CELL ********************

def region_mapping(df, REGION_MAPPING_TABLE_NAME):
    print("Reading the Sales Report, sales office dimension table for region/hub mapping")
    ## Read the current Sales Office to Region/Hub mapping available in the Sales Report lakehouse
    #SO_df = spark.sql("SELECT * FROM `BSI - Marketing`.Sales_Report.dbo.dim_sales_office")
    SO_df = spark.sql(f"SELECT * FROM {REGION_MAPPING_TABLE_NAME}")

    ## Select the Sales Office & Hub columns and transform to enable merging w/ Actual Sales Data
    SO_df = SO_df.select("Sales_Office", "Hub")
    SO_df = SO_df.withColumn("SO_test", trim(split(col("Sales_Office"),"-")[1]))
    SO_df = SO_df.select("SO_test", "Hub").distinct()

    ## Merge w/ the Actual Sales Data
    merged_df = df.join(SO_df, col("Sales_Office") == col("SO_test"),  how="left")
    print(f"Intial raw data had {df.count()} rows")
    print(f"The new merged dataframe has {merged_df.count()} rows")
    print(f"the initial minus the merged dataframe is a difference of {df.count()-merged_df.count()}")
    if df.count()-merged_df.count()!=0:
        raise ValueError("ERROR: There is a mismatch with the original dataframe size vs the current post merge size")

    ## dropping unneeded merge column
    merged_df = merged_df.drop("SO_test")

    ## Manual mapping due to these 2 values not mapping correctly with the mapping from the Sales Office table
    merged_df = merged_df.withColumn("Hub", when(col("Sales_Office")=="BITZER PAKISTAN(SMC-PRIVATE)LTD","EMEA").otherwise(col("Hub")))\
                            .withColumn("Hub", when(col("Sales_Office")=="SO2000", "S.America").otherwise(col("Hub")))

    if merged_df.filter(col("Hub").isNull()).count() > 0:
        raise ValueError("ERROR: There are records with unmapped Hub / Regions that need to be mapped")
    else:
        print("All records have been correctly mapped to a Hub / Region and there are no null values present")

    ## change the naming of "Hub" column to "Region"
    merged_df = merged_df.withColumnRenamed("Hub", "Region")

    return merged_df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## STEP 1: Raw -> Filtered Data

# CELL ********************

## Load raw data Actuals data
df = load_actuals_from_lakehouse(RAW_TABLE_NAME, _FABRIC_COLUMN_RENAME, _FABRIC_FILTER_COLS)

## Complete filtering in order to extract only the relevant actuals data 
    ## Cast "Value" column to double as it was originally a string
df = preprocess_actuals_data(df, PRODUCT_CATEGORIES)
df = df.withColumn("Value", col("Value").cast("double"))

print("*"*80)

merged_df = region_mapping(df, REGION_MAPPING_TABLE_NAME)


# filtering out the most recent months data as it is only partially complete and can mess up training data
max_date = merged_df.select(trunc(max("Date"), "month").alias("max_date")).first()["max_date"]
print(f'the most recent month of data is {max_date}, and all data from this month will be excluded')
merged_df = merged_df.where(col("Date")<max_date)

## write cleaned and filtered data to the lakehouse
write_to_lakehouse(merged_df, FILTERED_TABLE_NAME)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # STEP 2: Filtered Data -> Aggregated Data for Models

# MARKDOWN ********************

# #### group_data method

# CELL ********************

def group_data(df, group_cols, target_col):
    if group_cols:
        grouped_df = df.groupBy(*group_cols+["Date"]).agg(sum(target_col).alias(target_col))
    else:
        grouped_df = df.groupBy("Date").agg(sum(target_col).alias(target_col))

    return grouped_df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Fill_Data Method

# CELL ********************

def fill_data(df, group_cols, target_col):
    """
    Create a complete monthly time series for every group using the
    global minimum and maximum dates across the dataset.

    Missing target values are filled with 0.
    """

    # ---------------------------------------------------------
    # Global date range
    # ---------------------------------------------------------
    min_date, max_date = (
        df.selectExpr(
            "min(Date) as min_date",
            "max(Date) as max_date"
        )
        .first()
    )

    # ---------------------------------------------------------
    # Monthly calendar
    # ---------------------------------------------------------
    date_df = (
        spark.range(1)
        .select(
            explode(
                sequence(
                    lit(min_date),
                    lit(max_date),
                    expr("INTERVAL 1 MONTH")
                )
            ).alias("Date")
        )
    )

    # ---------------------------------------------------------
    # Distinct groups
    # ---------------------------------------------------------
    groups_df = df.select(*group_cols).distinct()

    # ---------------------------------------------------------
    # Complete group × month grid
    # ---------------------------------------------------------
    full_grid = groups_df.crossJoin(date_df)

    # ---------------------------------------------------------
    # Join original data
    # ---------------------------------------------------------
    filled_df = (
        full_grid
        .join(df, on=group_cols + ["Date"], how="left")
        .withColumn(
            target_col,
            coalesce(col(target_col), lit(0.0))
        )
    )

    # ---------------------------------------------------------
    # Return ordered dataframe
    # ---------------------------------------------------------
    return (
        filled_df
        .orderBy(*group_cols, "Date")
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Reading data from filtered base data

# CELL ********************

filtered_actuals = spark.read.table(FILTERED_TABLE_NAME)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## writing raw and filled topline/middle data to tables

# CELL ********************

final_df = group_data(filtered_actuals, GRP_COLS, TARGET_COL)
final_df = final_df.withColumn('series', concat_ws("___",*GRP_COLS))
write_to_lakehouse(final_df, raw_act_table)

filled_df = fill_data(final_df, GRP_COLS, TARGET_COL)
filled_df = filled_df.withColumn('series', concat_ws("___",*GRP_COLS))
write_to_lakehouse(filled_df, init_act_table)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Applying the Historical Data Cutoff by Product Category

# CELL ********************

act_data = spark.read.table(init_act_table)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## creating dataframe w/ the cutoff dates
data = []
for key, item in HISTORICAL_CUTOFF_DATES.items():
    data.append({
                "series":key,
                "cutoff_date":item
                })
df = pd.DataFrame(data)
cutoff_dates = spark.createDataFrame(df)
cutoff_dates = cutoff_dates.withColumn("cutoff_date", to_date(col("cutoff_date"), "dd-MM-yyyy"))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

filtered_data = (
    act_data.alias("d")
    .join(
        cutoff_dates.alias("c"),
        col("d.Product_Category").startswith(col("c.series")),
        "inner"
    )
    .filter(col("d.Date")>=col("c.cutoff_date"))
    .select("d.*")
)
display(filtered_data.limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

filtered_data.write.format("delta").mode("overwrite").saveAsTable(cutoff_data_table)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
