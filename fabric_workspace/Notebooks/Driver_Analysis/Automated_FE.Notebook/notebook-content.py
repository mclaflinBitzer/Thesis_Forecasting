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

#original drivers
compiled_drivers = spark.read.table(driver_table).select("Country","Indicator","Region","Date","Value")

# actual data
data = spark.read.table(actuals_table)
data = data.groupBy(*ACT_GRP_COLS,"Date").agg(sum("Quantity").alias("Quantity"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Driver Processing

# CELL ********************

aggregated_drivers = compiled_drivers.groupBy(*DRV_GRP_COLS,'Date').agg(sum('Value').alias('Value'))

if 'Indicator' in DRV_GRP_COLS and len(DRV_GRP_COLS) > 1:
    world_agg = compiled_drivers.groupBy('Indicator','Date').agg(sum("Value").alias("Value"))
    
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

def feature_engineering(df, drv_grp_cols):
    ## Feature Transformations to do
        ## levels
        ## Rolling Mean / STD (3,6,12)
        ## Growth: YoY, MoM, diff

    ## levels
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
# - corr clustering removal
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

def reformatting_data(df, grp_cols):
    id_cols = ["Date"]
    feature_cols = [c for c in df.columns if c not in [*grp_cols, "Date"]]

    df_long = (
        df
        .withColumn(
            "feature",
            explode(
                array(*[
                    struct(
                        concat_ws(
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

display(data.select("Product_Category").distinct())
display(data.select("Region").distinct())

data = data.filter(
    (col("Product_Category")=="ALU") &
    (col("Region")=="China")
)

display(data.select("Product_Category", "Region").distinct())

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

    conditions = []

    for c in join_pairs:
        # Automatically map target_* -> feature_*
        feature_col = c

        conditions.append(
            col(f"{target_alias}.{c}") == col(f"{feature_alias}.{feature_col}")
        )


    # Always join on the dates
    conditions.append(
        col(f"{target_alias}.Date") ==
        col(f"{feature_alias}.Date")
    )

    join_cond = reduce(operator.and_, conditions)



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
    window = Window.partitionBy(*cols)
    df = df.withColumn("abs_corr", abs(col("Correlation")))
    df = df.withColumn("max_corr", max(col("abs_corr")).over(window))


    ## filter out records with a correlations < .3 or Indicator is NaN
    df_filtered = df.filter(
        (col("max_corr") > 0.15) &
        (col("max_corr") < .95) &
        (col("Indicator").isNotNull()) &
        (col("Indicator") != "NaN") &
        (col('coverage') >= 0.8) &
        (col('n_overlap') >= 24)
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

## APPLICATION OF CCF 

split_col = split(col("Feature"), "__")
feature_serie_cols = DRV_GRP_COLS + ['Feature_name']

for i, c in enumerate(feature_serie_cols):
    feature_residuals = feature_residuals.withColumn(c, split_col.getItem(i))


# data_residuals = data_residuals.withColumnsRenamed(ACT_COLS_RENAME)
# feature_residuals = feature_residuals.withColumnsRenamed(DRV_COLS_RENAME)



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

display(filtered_features.limit(20))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## melt expanded features to a long format\

melt_cols = DRV_GRP_COLS.copy()
melt_cols.remove("Feature_name")


feature_value_cols = [
    "level",
    "rolling_mean_3", "rolling_std_3",
    "rolling_mean_6", "rolling_std_6",
    "rolling_mean_12", "rolling_std_12",
    "rolling_mean_18", "rolling_std_18",
    "rolling_mean_24", "rolling_std_24",
    "diff", "YoY_pct", "MoM_pct",
]

stack_expr = "stack({}, {}) as (Feature_name, Feature_value)".format(
    len(feature_value_cols),
    ", ".join(f"'{c}', {c}" for c in feature_value_cols)
)

expanded_features_long = expanded_features.select(
    *melt_cols, "Date", expr(stack_expr)
)


grp_cols = list(dict.fromkeys(DRV_GRP_COLS + ACT_GRP_COLS))

combos = filtered_features.select(*grp_cols, "Lag").distinct()

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

conditions = []
for col_name in DRV_GRP_COLS:
    conditions.append(
        col(f"t.{col_name}")== col(f"f.{col_name}")
    )
conditions.append(
    col("t.driver_date") == col("f.Date")
)


result = (
    target_w_features.alias("t")
    .join(
        expanded_features_long.alias("f"),
        join_cond,
        "left"
    ).select("t.*", "f.Feature_value")
)



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
    if c not in non_scaled_cols + [target_col]
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



## FEATURE IMPORTANCE
coefficients = list(enet_model.coefficients)

feature_impact = (
    spark.createDataFrame(
        zip(feature_cols, coefficients),
        ["Feature", "Coefficient"]
    )
    .withColumn("AbsImpact", abs("Coefficient"))
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

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
