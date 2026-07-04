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
from pyspark.sql.types import *
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import ccf 
import operator
from functools import reduce
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from functools import reduce

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## TOPLINE
T_series = ['series']
T_DRV_GRP_COLS = ['Indicator']
T_ACT_GRP_COLS = ['Product_Category','series']

T_cols = T_ACT_GRP_COLS + T_DRV_GRP_COLS

T_ACT_COLS_RENAME = {"Date":"target_date","residual":"target_residual"}  
T_DRV_COLS_RENAME = {"Date":"feature_date","residual":"feature_residual"}

T_ACT_COLS_RN = ['Product_Category', 'series']
T_DRV_COLS_RN = ['Indicator']

T_actuals_table = "Sales_Forecasting.silver.topline_cutoff_data"

## MIDDLE
M_series = ['series']
M_DRV_GRP_COLS = ['Region','Indicator']
M_ACT_GRP_COLS = ['Product_Category', 'Region','series']

M_cols = M_ACT_GRP_COLS + M_DRV_GRP_COLS

M_ACT_COLS_RENAME = {"Date":"target_date","residual":"target_residual","Region":"target_region"}  
M_DRV_COLS_RENAME = {"Date":"feature_date","residual":"feature_residual","Region":"feature_region"}

M_ACT_COLS_RN = ['Product_Category', 'target_region', 'series']
M_DRV_COLS_RN = ['feature_region','Indicator']

M_actuals_table = "Sales_Forecasting.silver.middle_cutoff_data"


## shared
col_renamed = {"Quantity":"target_value","Value":"feature_value"}
driver_table = "Sales_Forecasting.silver.compiled_drivers"


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

Topline = False

if Topline:
    series = T_series
    DRV_GRP_COLS = T_DRV_GRP_COLS 
    ACT_GRP_COLS = T_ACT_GRP_COLS 

    level_cols = list(dict.fromkeys(T_cols)) 
    actuals_table = T_actuals_table 

    ACT_COLS_RENAME = T_ACT_COLS_RENAME
    DRV_COLS_RENAME = T_DRV_COLS_RENAME

    ACT_COLS_RN = T_ACT_COLS_RN
    DRV_COLS_RN = T_DRV_COLS_RN

else:
    series = M_series 
    DRV_GRP_COLS = M_DRV_GRP_COLS 
    ACT_GRP_COLS = M_ACT_GRP_COLS 

    level_cols = list(dict.fromkeys(M_cols))
    actuals_table = M_actuals_table

    ACT_COLS_RENAME = M_ACT_COLS_RENAME
    DRV_COLS_RENAME = M_DRV_COLS_RENAME

    ACT_COLS_RN = M_ACT_COLS_RN
    DRV_COLS_RN = M_DRV_COLS_RN

## shared
col_renamed = {"Quantity":"target_value","Value":"feature_value"}
driver_table = "Sales_Forecasting.silver.compiled_drivers"


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Reading Data

# CELL ********************

# ORIGINAL / BASE LEVEL DRIVERS
compiled_drivers = spark.read.table(driver_table).select("Country","Indicator","Region","Date","Value")

# ACTUALS TABLE
data = spark.read.table(actuals_table)
## aggregating based on the ACT GRP COLS defined
data = data.groupBy(*ACT_GRP_COLS,"Date").agg(sum("Quantity").alias("Quantity"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Driver Processing

# CELL ********************

## Aggregating the drivers based upon the DRV GRP COLS defined 
aggregated_drivers = compiled_drivers.groupBy(*DRV_GRP_COLS,'Date').agg(sum('Value').alias('Value'))

# if Indicator is the only value in DRV GRP COLS then the data will be aggregated at a lower level than world
    # this ensures that "WORLD" level aggregated drivers are added to the aggregated_drivers set

if 'Indicator' in DRV_GRP_COLS and len(DRV_GRP_COLS) > 1:
    # creates world level drivers
    world_agg = compiled_drivers.groupBy('Indicator','Date').agg(sum("Value").alias("Value"))
    
    # populates the other DRV GRP COLS defined that aren't "Indicator" with the value of "World"
    for cols in DRV_GRP_COLS:
        if cols!='Indicator':
            world_agg = world_agg.withColumn(cols, lit("World"))
            print(f"{cols} added using .withColumn, populated with lit(World)")

    aggregated_drivers = aggregated_drivers.unionByName(world_agg)
    
else:
    print('world agg already done')


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Feature Engineering / Expansion

# CELL ********************

# automatically creating versions of base drivers using pre defined features/windows

def feature_engineering(df, drv_grp_cols):
    ## Feature Transformations to do
        ## levels
        ## Rolling Mean / STD (3,6,12)
        ## Growth: YoY, MoM, diff

    ## creation of levels col / original driver values
    df = df.withColumn("level", col("Value")).drop("Value")

    ## Window creation
    w = Window.partitionBy(*drv_grp_cols).orderBy("Date")
    w_3 = Window.partitionBy(*drv_grp_cols).orderBy("Date").rowsBetween(-2,0)
    w_6 = Window.partitionBy(*drv_grp_cols).orderBy("Date").rowsBetween(-5,0)
    w_12 = Window.partitionBy(*drv_grp_cols).orderBy("Date").rowsBetween(-11,0)
    w_18 = Window.partitionBy(*drv_grp_cols).orderBy("Date").rowsBetween(-17,0)
    w_24 = Window.partitionBy(*drv_grp_cols).orderBy("Date").rowsBetween(-23,0)


    windows = {
        3: w_3,
        6: w_6,
        12: w_12,
        18: w_18,
        24: w_24
    }

    

    ## Rolling Mean & Stddev
        ## leveraging the widows dict in order to loop feature creation across windows
    for size, win in windows.items():
        df = df.withColumn(f"rolling_mean_{size}", avg("level").over(win))
        df = df.withColumn(f"rolling_std_{size}", stddev("level").over(win))


    ## Growth: YoY, MoM, Diff
    df = df.withColumn("diff", (col("level") - lag("level", 1).over(w)))
    df = df.withColumn("YoY_pct", 
                            when(lag("level",12).over(w) == 0, None)
                            .otherwise((col("level")/lag("level",12).over(w)) -1))
    df = df.withColumn("MoM_pct",
                            when(lag("level",1).over(w)==0, None)
                            .otherwise((col("level") / lag("level", 1).over(w))-1))

    return df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# applying feature_engineeringh method in order to expand the feature set
expanded_features = feature_engineering(aggregated_drivers, DRV_GRP_COLS)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Classical Feature Processing
# - STL residual application
# - correlation analysis
# -- filter out low / too high correlations
# - ccf w/ lag identification
# - Elastic Net selection
#         
#         - scale data for elastic net
#         - select features based output per target series
# 
# - Finalize prior to Model ingestion

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

display(expanded_features.limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def reformatting_data(df, grp_cols):
    ## grp cols is the driver level aggregation (DRV GRP COLS)
    ## id col of date determines the alignment/grouping on date
    ## feature cols collects all columns that aren't defined within the id or grp cols (ex level, rolling mean, rolling std, etc)
    id_cols = ["Date"]
    feature_cols = [c for c in df.columns if c not in [*grp_cols, "Date"]]

    df_long = (
        df
        .withColumn(
            "feature",
            explode(
                array(*[
                    struct(                 ## leverage struct to create individual mapping between the feature col and the value
                        concat_ws(          ## leverage concat_ws to create a "feature" column comprised of the DRV GRP COL values
                            "__", 
                            *[col(g) for g in grp_cols],
                             lit(c)).alias("Feature"),
                        col(c).cast("double").alias("Value")
                    )
                    for c in feature_cols
                ])
            )
        )
        .select(
            "Date",
            col("feature.Feature").alias("Feature"),
            col("feature.Value").alias("Value")
        )
    )

    return df_long

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # TEMP DATA Filter to manage scale

# CELL ********************

data = data.filter(
    (col("Product_Category")=="ALU") &
    (col("Region")=="China")
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

reformatted_df = reformatting_data(expanded_features, DRV_GRP_COLS)


## FEATURE RESIDUALS

feature_stl_schema = StructType([
    StructField("Feature", StringType(), False),
    StructField("Date", DateType(), False),
    StructField("Value", DoubleType(), True),
    StructField("residual", DoubleType(), True)
])

feature_residuals = reformatted_df.groupBy("Feature").applyInPandas(stl_decompose, schema=feature_stl_schema)



## DATA RESIDUALS

data_stl_schema = StructType(
    [StructField(c, StringType(), False) for c in ACT_GRP_COLS] +
    [
        StructField("Date", DateType(), False),
        StructField("Value", DoubleType(), True),
        StructField("residual", DoubleType(), True)
    ]

)

## renaming quantity col of actuals to 'value' for stl processing 
data = data.withColumnRenamed("Quantity","Value")

data_residuals = data.groupBy(*ACT_GRP_COLS).applyInPandas(stl_decompose, schema=data_stl_schema)

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

    # dynamically creates join conditions of t.col_name == f.col_name for the join between target and feature residuals
    conditions = []

    for c in join_pairs:
        # Automatically map target_* -> feature_*
        feature_col = c

        conditions.append(
            col(f"{target_alias}.{c}") == col(f"{feature_alias}.{feature_col}")
        )

    # Appending "Date" join condition
    conditions.append(
        col(f"{target_alias}.Date") ==
        col(f"{feature_alias}.Date")
    )

    join_cond = reduce(operator.and_, conditions)


    # left join on targets using join_conditions
    # selecting all columns in the original target df (renaming t.residual to target_residual)
    # only selecting the residual col from the feature df and renaming to feature_residual
    return (
        t.join(f, join_cond, "left")
        .select(
            *[
                col(f"{target_alias}.{c}").alias(
                    "target_residual" if c == "residual" else c
                )
                for c in target_df.columns
            ],
            col(f"{feature_alias}.residual").alias("feature_residual")
        )
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def apply_ccf(df):
    pdf = df.sort_values('Date')

    result_cols = [c for c in pdf.columns if c not in ['Date','Value','feature_residual','target_residual']]

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

# CELL ********************

def ccf_filtering(df, cols, serie_col):
    # cols is a list of the ACT GRP COLS and the unique/serie columns for drivers
    ## serie_col is the 'series' column from the actuals data

    window = Window.partitionBy(*cols)
    df = df.withColumn("abs_corr", abs(col("Correlation")))
    df = df.withColumn("max_corr", max(col("abs_corr")).over(window))


    ## filterng out features from down stream processing based on the following conditions
    df_filtered = df.filter(
        (col("max_corr") > 0.15) &
        (col("max_corr") < .95) &
        (col("Indicator").isNotNull()) &
        (col("Indicator") != "NaN") &
        (col('coverage') >= 0.8) &
        (col('n_overlap') >= 24)
    )

    # creating a max_corr for each unique ACT GRP and DRIVER combination
    df_ranked = df_filtered.groupBy(*cols).agg(first(col("max_corr")).alias("max_corr"))
    
    
    if len(serie_col) == 0:
        w = Window.orderBy(desc('max_corr'))
    else:
        w = Window.partitionBy(*serie_col).orderBy(desc("max_corr"))

    # creating a rank value for the max_corr values of features within each ACT serie partition
    df_ranked = df_ranked.withColumn("rank", rank().over(w))
    df_rank_filtered = df_ranked.filter(col("rank")<=500)

    ## only keeping features that pass both of the selection/filtering approaches
    df_final_filtered = df_rank_filtered.join(df_filtered, [*cols], 'inner').drop(df_rank_filtered['max_corr'],df_rank_filtered['rank'])

    return df_final_filtered

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## APPLICATION OF CCF 

split_col = split(col("Feature"), "__")
feature_serie_cols = DRV_GRP_COLS + ['Feature_name']

for i, c in enumerate(feature_serie_cols):
    feature_residuals = feature_residuals.withColumn(c, split_col.getItem(i))




## CREATE MAPPING FOR TARGET-DRIVER COMBINATIONS
DRV_GRP_COLS = DRV_GRP_COLS + ["Feature_name"]
data_distinct = data_residuals.select(*ACT_GRP_COLS).distinct()
drv_distinct = feature_residuals.select(*feature_serie_cols).distinct()



join_col = []
for a_col in ACT_GRP_COLS:
    for d_col in feature_serie_cols:
        if a_col == d_col:
            join_col.append(d_col)
            print(f"{d_col} added to join col list")

if len(join_col) == 0:
    print("no shared columns so cross join was done")
    pairs = data_distinct.crossJoin(drv_distinct)
else:
    print(f"shared columns so the join was done on {join_col}")
    pairs = data_distinct.join(drv_distinct, join_col, 'inner')


if len(ACT_GRP_COLS) == 0:
    data_expanded = data_residuals.crossJoin(broadcast(pairs))
else:
    data_expanded = data_residuals.join(broadcast(pairs), [*ACT_GRP_COLS], 'inner')


features_expanded = broadcast(pairs).join(feature_residuals, [*feature_serie_cols], 'inner')


join_cols = list(dict.fromkeys(ACT_GRP_COLS + feature_serie_cols))

joined_data_features = join_target_feature(
    data_expanded, features_expanded, join_cols
)

ccf_schema = StructType(
    [StructField(c, StringType(), False) for c in join_cols] +
    [
        StructField("Lag", IntegerType(), False),
        StructField("Correlation", DoubleType(), True),
        StructField('coverage', DoubleType(), True),
        StructField('n_overlap', IntegerType(), False)
    ]
)

ccf_output = joined_data_features.groupBy(join_cols).applyInPandas(apply_ccf, schema=ccf_schema)

ccf_filtered = ccf_filtering(ccf_output, join_cols, series)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def top_features(ccf_filtered, DRV_GRP_COLS, ACT_GRP_COLS, num_features):
    
    ## select the top x features per ACT group & select their ideal lag
    grp_cols = list(dict.fromkeys(DRV_GRP_COLS + ACT_GRP_COLS))

    ## create rec_lag flag and stable flag
    w = Window.partitionBy(*grp_cols).orderBy("Lag").rowsBetween(-1,1)
    ccf_filtered = (
        ccf_filtered
        .withColumn("rec_lag", when(col("abs_corr")==col("max_corr"), lit(1)).otherwise(lit(0)))
        .withColumn("rolling_corr_avg", mean(col("Correlation")).over(w))
        .withColumn("rolling_corr_std", stddev(col("Correlation")).over(w))
        .withColumn("stable_flag",
            when(
                (col("Correlation") >= col("rolling_corr_avg") - col("rolling_corr_std")) &
                (col("Correlation") <= col("rolling_corr_avg") + col("rolling_corr_std")),
                lit(1)
            ).otherwise(lit(0))
        )
    )

    ## filtering for only records with the rec lag and stable period flags as true
    ccf_filtered = ccf_filtered.filter(
        (col("rec_lag")==1) & (col("stable_flag")==1)
    )


    w_a = Window.partitionBy(*ACT_GRP_COLS).orderBy(desc("max_corr"))

    ccf_filtered = ccf_filtered.withColumn("rank", dense_rank().over(w_a))

    filtered_top = ccf_filtered.filter(col("rank")<=num_features)

    return filtered_top

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

filtered_features = top_features(ccf_filtered, DRV_GRP_COLS, ACT_GRP_COLS, 50)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## melt expanded features to a long format

# the 'melt_cols' are 'grouping cols' or the columns to remain unchanged 
melt_cols = DRV_GRP_COLS.copy()
melt_cols.remove("Feature_name")

# all of the feature columns outputed from the feature_engineering method
feature_value_cols = [
    "level",
    "rolling_mean_3", "rolling_std_3",
    "rolling_mean_6", "rolling_std_6",
    "rolling_mean_12", "rolling_std_12",
    "rolling_mean_18", "rolling_std_18",
    "rolling_mean_24", "rolling_std_24",
    "diff", "YoY_pct", "MoM_pct",
]

## unpivoting the data (taking the wide format of column per feature and it's value to long feature_name & feature_value columns)
expanded_features_long = expanded_features.unpivot(
    ids=melt_cols + ["Date"],
    values=feature_value_cols,              ## the columns to unpivot from wide to long format
    variableColumnName="Feature_name",      ## column name of the original wide columns variable (col name)
    valueColumnName="Feature_value"         ## the column name of the original value within the wide columns 
)



grp_cols = list(dict.fromkeys(DRV_GRP_COLS + ACT_GRP_COLS))

combos = filtered_features.select(*grp_cols, "Lag").distinct()

# combining the actuals original data w/ the filtered features and their identified lags
# creating the driver_date column which is the actuals data lagged by the features identified lag
target_w_features = (
    data
    .join(combos, ACT_GRP_COLS, "inner")
    .withColumn("driver_date", expr("add_months(Date, -Lag)"))
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## creating join conditions based on the DRV GRP COLS

conditions = []
for col_name in DRV_GRP_COLS:
    conditions.append(
        col(f"t.{col_name}")== col(f"f.{col_name}")
    )

# appending "date" as a join condition
conditions.append(
    col("t.driver_date") == col("f.Date")
)

## creating the actual conditions
join_cond = reduce(operator.and_, conditions)

## joining the target data w/ the driver/feature utilizing the DRV GRP COLS and driver_date (lagged date of feature) with the drivers value at that date
result = (
    target_w_features.alias("t")
    .join(
        expanded_features_long.alias("f"),
        join_cond,
        "left"
    ).select("t.*", "f.Feature_value")
)

display(result.limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

feature_matrix = (
    result
    .withColumn("feature_col", concat_ws("__",*DRV_GRP_COLS, col("Lag").cast("string")))
    .groupBy(*ACT_GRP_COLS, "Date","Value")
    .pivot("feature_col")
    .agg(first("Feature_value"))
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(feature_matrix.orderBy("series",asc("Date")))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Elastic Net Prep

# CELL ********************

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## getting required col mapping for scaling
id_cols = ACT_GRP_COLS + ["Date"]
target_col = "Value"
feature_cols = [
    c for c in feature_matrix.columns 
    if c not in id_cols + [target_col]
]

# removing records without target
train_df = (
    feature_matrix
    .filter(col(target_col).isNotNull())
)

## fill missing feature values
train_df = train_df.fillna(0, subset=feature_cols)

## Assemble feature vector
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol='features',
    handleInvalid='keep'
)

assembled_df = assembler.transform(train_df)


## Standardize / Scale values
scaler = StandardScaler(
    inputCol='features',
    outputCol='scaled_features',
    withMean=True,
    withStd=True
)

scaler_model = scaler.fit(assembled_df)

scaled_df = scaler_model.transform(assembled_df)



## ELASTIC NET
elastic_net = LinearRegression(
    labelCol=target_col,
    featuresCol='scaled_features',

    ## elastic net parameters
    elasticNetParam=0.5,
    regParam=0.1,
    maxIter=1000,
    tol=1e-6
)

enet_model = elastic_net.fit(scaled_df)


## MODEL SUMMARY
summary = enet_model.summary
print(f"R²      : {summary.r2:.4f}")
print(f"RMSE    : {summary.rootMeanSquaredError:.4f}")
print(f"MAE     : {summary.meanAbsoluteError:.4f}")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## FEATURE IMPORTANCE

schema = StructType([
    StructField("Feature", StringType(), False),
    StructField("Coefficient", DoubleType(), False)
])

feature_impact = (
    spark.createDataFrame(
        [(f, float(c)) for f, c in zip(feature_cols, enet_model.coefficients)],
        schema=schema
    )
    .withColumn("AbsImpact", abs(col("Coefficient")))
    .orderBy(desc("AbsImpact"))
)

display(feature_impact)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print(feature_cols[0])


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

feature_matrix.select(feature_cols[0]).show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## potential code for elasticnetCV

# CELL ********************

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler as SkScaler
import pandas as pd
import numpy as np
import builtins

# ============================================================
# CONFIG
# ============================================================
target_col = "Value"
id_cols = ACT_GRP_COLS + ["Date"]
feature_cols = [c for c in feature_matrix.columns if c not in id_cols + [target_col]]

MIN_HISTORY = 24          # minimum months of target history required to fit
TEST_FRAC = 0.2            # holdout fraction, taken from the END (time-respecting)
CORR_THRESHOLD = 0.90       # drop near-duplicate engineered features pre-fit
N_BOOT = 20                 # bootstrap resamples for stability selection
BOOT_SAMPLE_FRAC = 0.8
STABILITY_THRESHOLD = 0.6   # keep features selected in >=60% of bootstraps
MAX_FEATURES_OUT = 25        # cap on final selected features per series

# ============================================================
# OUTPUT SCHEMAS
# ============================================================
diagnostics_schema = StructType(
    [StructField(c, StringType(), False) for c in ACT_GRP_COLS] +
    [
        StructField("Feature", StringType(), False),
        StructField("Coefficient", DoubleType(), True),
        StructField("abs_coef", DoubleType(), True),
        StructField("stability_score", DoubleType(), True),
        StructField("selected", IntegerType(), True),
        StructField("best_alpha", DoubleType(), True),
        StructField("best_l1_ratio", DoubleType(), True),
        StructField("test_r2", DoubleType(), True),
        StructField("n_obs", IntegerType(), True),
    ]
)

# ============================================================
# CORE FIT FUNCTION — runs once per ACT_GRP_COLS group, distributed via Spark
# ============================================================
def fit_and_select_features(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf = pdf.sort_values("Date").reset_index(drop=True)
    id_vals = {c: pdf[c].iloc[0] for c in ACT_GRP_COLS}

    def empty_result(reason_cols=None):
        cols = feature_cols if reason_cols is None else reason_cols
        return pd.DataFrame([{
            **id_vals, "Feature": f, "Coefficient": None, "abs_coef": None,
            "stability_score": None, "selected": 0, "best_alpha": None,
            "best_l1_ratio": None, "test_r2": None, "n_obs": len(pdf),
        } for f in cols])

    X = pdf[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(pdf[target_col], errors="coerce")

    valid = y.notna()
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)

    if len(y) < MIN_HISTORY:
        return empty_result()

    # ---- drop near-duplicate engineered features (e.g. rolling_mean_3 vs rolling_mean_6) ----
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > CORR_THRESHOLD)]
    kept_cols = [c for c in X.columns if c not in to_drop]
    X = X[kept_cols]

    if X.shape[1] == 0:
        return empty_result(kept_cols)

    # ---- time-respecting split: last TEST_FRAC as holdout, never shuffled ----
    split_idx = int(len(y) * (1 - TEST_FRAC))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(y_train) < 12 or len(y_test) < 3:
        return empty_result(kept_cols)

    # ---- median impute from TRAIN ONLY (no leakage) ----
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    scaler = SkScaler().fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=kept_cols)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=kept_cols)

    n_splits = builtins.min(5, builtins.max(2, len(y_train) // 12))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    try:
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
            alphas=np.logspace(-3, 1, 30),
            cv=tscv,
            max_iter=10000,
            tol=1e-6,
        ).fit(X_train_s, y_train)
    except Exception:
        return empty_result(kept_cols)

    test_r2 = model.score(X_test_s, y_test)

    # ---- bootstrap stability selection on train fold ----
    stability_counts = pd.Series(0, index=kept_cols, dtype=float)
    for _ in range(N_BOOT):
        boot_idx = X_train_s.sample(frac=BOOT_SAMPLE_FRAC, replace=True).index
        try:
            m = ElasticNetCV(
                l1_ratio=[model.l1_ratio_], alphas=[model.alpha_],
                cv=2, max_iter=5000,
            ).fit(X_train_s.loc[boot_idx], y_train.loc[boot_idx])
            stability_counts += (pd.Series(m.coef_, index=kept_cols) != 0).astype(int)
        except Exception:
            continue
    stability_score = stability_counts / N_BOOT

    coefs = pd.Series(model.coef_, index=kept_cols)
    selected = (stability_score >= STABILITY_THRESHOLD) & (coefs != 0)

    out = pd.DataFrame({
        **{c: id_vals[c] for c in ACT_GRP_COLS},
        "Feature": kept_cols,
        "Coefficient": coefs.values,
        "abs_coef": coefs.abs().values,
        "stability_score": stability_score.values,
        "selected": selected.astype(int).values,
        "best_alpha": model.alpha_,
        "best_l1_ratio": model.l1_ratio_,
        "test_r2": test_r2,
        "n_obs": len(pdf),
    })

    # cap final selection to top MAX_FEATURES_OUT by |coefficient| among selected
    out = out.sort_values("abs_coef", ascending=False)
    keep_mask = out["selected"] == 1
    if keep_mask.sum() > MAX_FEATURES_OUT:
        drop_idx = out[keep_mask].index[MAX_FEATURES_OUT:]
        out.loc[drop_idx, "selected"] = 0

    return out

# ============================================================
# RUN — distributed across Spark executors, one fit per series
# ============================================================
feature_diagnostics = (
    feature_matrix
    .filter(col(target_col).isNotNull())
    .groupBy(*ACT_GRP_COLS)
    .applyInPandas(fit_and_select_features, schema=diagnostics_schema)
)

feature_diagnostics.cache()
display(feature_diagnostics.orderBy(*ACT_GRP_COLS, desc("abs_coef")))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

selected_features = (
    feature_diagnostics
    .filter(col("selected") == 1)
    .select(*ACT_GRP_COLS, "Feature")
)

# long format: one row per (series, Date, selected Feature, its value)
feature_matrix_long = melt_features_long(  # reuse your existing melt helper, or:
    feature_matrix, ACT_GRP_COLS, feature_cols
) if 'melt_features_long' in dir() else (
    feature_matrix.select(
        *ACT_GRP_COLS, "Date", target_col,
        expr(f"stack({len(feature_cols)}, " +
             ", ".join(f"'{c}', `{c}`" for c in feature_cols) +
             ") as (Feature, Feature_value)")
    )
)

# keep only rows whose Feature was selected for that specific series
trimmed_long = feature_matrix_long.join(
    selected_features, [*ACT_GRP_COLS, "Feature"], "inner"
)

# pivot back to wide — final downstream-model-ready matrix
final_feature_matrix = (
    trimmed_long
    .groupBy(*ACT_GRP_COLS, "Date", target_col)
    .pivot("Feature")
    .agg(first("Feature_value"))
)

display(final_feature_matrix.limit(50))

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

# CELL ********************

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler as SkScaler
import pandas as pd
import numpy as np
import builtins

result_schema = StructType(
    [StructField(c, StringType(), False) for c in ACT_GRP_COLS] +
    [
        StructField("Feature", StringType(), False),
        StructField("Coefficient", DoubleType(), False),
        StructField("abs_coef", DoubleType(), False),
        StructField("best_alpha", DoubleType(), False),
        StructField("best_l1_ratio", DoubleType(), False),
        StructField("test_r2", DoubleType(), True),
    ]
)

def fit_enet_per_series(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf = pdf.sort_values("Date")
    id_vals = {c: pdf[c].iloc[0] for c in ACT_GRP_COLS}

    feature_cols = [c for c in pdf.columns if c not in ACT_GRP_COLS + ["Date", target_col]]
    X = pdf[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(pdf[target_col], errors="coerce")

    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    if len(y) < 24 or X.shape[1] == 0:   # not enough history to fit safely
        return pd.DataFrame([{**id_vals, "Feature": None, "Coefficient": None,
                               "abs_coef": None, "best_alpha": None,
                               "best_l1_ratio": None, "test_r2": None}])

    # time-respecting split: last 20% as holdout, never shuffled
    split_idx = int(len(y) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # median impute (fit on train only, avoid leakage)
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    scaler = SkScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # blocked (non-shuffled) CV folds for time series
    n_splits = builtins.min(5, builtins.max(2, len(y_train) // 12))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    model = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
        alphas=np.logspace(-3, 1, 30),
        cv=tscv,
        max_iter=10000,
        tol=1e-6,
    ).fit(X_train_s, y_train)

    test_r2 = model.score(X_test_s, y_test)

    coefs = pd.Series(model.coef_, index=feature_cols)
    out = pd.DataFrame({
        **{c: id_vals[c] for c in ACT_GRP_COLS},
        "Feature": coefs.index,
        "Coefficient": coefs.values,
        "abs_coef": coefs.abs().values,
        "best_alpha": model.alpha_,
        "best_l1_ratio": model.l1_ratio_,
        "test_r2": test_r2,
    })
    return out

from sklearn.model_selection import TimeSeriesSplit

feature_selection_output = (
    feature_matrix
    .filter(col(target_col).isNotNull())
    .groupBy(*ACT_GRP_COLS)
    .applyInPandas(fit_enet_per_series, schema=result_schema)
)

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
