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
T_feature_diagnostics_file = "/lakehouse/default/Files/Automated_Driver_Analysis/topline_feature_diagnostics.xlsx"
T_selected_feature_file = "/lakehouse/default/Files/Automated_Driver_Analysis/topline_selected_features.xlsx"

T_xgboost_diagnostics_file = "/lakehouse/default/Files/Automated_Driver_Analysis/topline_xgboost_feature_diagnostics.xlsx"
T_xgboost_selected_feature_file = "/lakehouse/default/Files/Automated_Driver_Analysis/topline_xgboost_selected_feature.xlsx"


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
M_feature_diagnostics_file = "/lakehouse/default/Files/Automated_Driver_Analysis/middle_feature_diagnostics.xlsx"
M_selected_feature_file = "/lakehouse/default/Files/Automated_Driver_Analysis/middle_selected_features.xlsx"

M_xgboost_diagnostics_file = "/lakehouse/default/Files/Automated_Driver_Analysis/middle_xgboost_feature_diagnostics.xlsx"
M_xgboost_selected_feature_file = "/lakehouse/default/Files/Automated_Driver_Analysis/middle_xgboost_selected_features.xlsx"


## shared
col_renamed = {"Quantity":"target_value","Value":"feature_value"}
driver_table = "Sales_Forecasting.silver.compiled_drivers"


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

Topline = True

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

    feature_diagnostics_file = T_feature_diagnostics_file
    selected_feature_file = T_selected_feature_file 
    xgb_feature_diagnostics_file = T_xgboost_diagnostics_file 
    xgb_selected_feature_file = T_xgboost_selected_feature_file

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

    feature_diagnostics_file = M_feature_diagnostics_file
    selected_feature_file = M_selected_feature_file

    xgb_feature_diagnostics_file = M_xgboost_diagnostics_file 
    xgb_selected_feature_file = M_xgboost_selected_feature_file


## shared
col_renamed = {"Quantity":"target_value","Value":"feature_value"}
driver_table = "Sales_Forecasting.silver.compiled_drivers"
target_col = 'Value'

elasticnet_init_features = 15
xgboost_init_features = 200

elasticnet_run = False
xgboost_run = True

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
data = spark.read.table(actuals_table).filter(col('Product_Category').isin('ALU',"SCREWS"))
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
    w_24 = Window.partitionBy(*drv_grp_cols).orderBy("Date").rowsBetween(-23,0)


    windows = {
        3: w_3,
        6: w_6,
        12: w_12,
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
expanded_features = feature_engineering(aggregated_drivers, DRV_GRP_COLS).cache()

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

def reformatting_data(df, drv_grp_cols):
    ## grp cols is the driver level aggregation (DRV GRP COLS)
    ## id col of date determines the alignment/grouping on date
    ## feature cols collects all columns that aren't defined within the id or grp cols (ex level, rolling mean, rolling std, etc)
    id_cols = ["Date"]
    feature_cols = [c for c in df.columns if c not in [*drv_grp_cols, "Date"]]

    df_long = (
        df
        .withColumn(
            "feature",
            explode(
                array(*[
                    struct(                 ## leverage struct to create individual mapping between the feature col and the value
                        concat_ws(          ## leverage concat_ws to create a "feature" column comprised of the DRV GRP COL values
                            "__", 
                            *[col(g) for g in drv_grp_cols],
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

def corr_clustering(df):

    # grouping columns are whatever isn't a measure
    group_cols = [
        c for c in df.columns
        if c not in [
            "target_date",
            "feature_id",
            "feature_residual"
        ]
    ]

    group_values = {
        c: df[c].iloc[0]
        for c in group_cols
    }

    wide_df = (
        df.pivot_table(
            index=group_cols + ["target_date"],
            columns="feature_id",
            values="feature_residual",
            aggfunc="first"
        )
        .reset_index()
    )

    X = (
        wide_df
        .drop(columns=group_cols + ["target_date"])
        .select_dtypes(include=[np.number])
        .fillna(0)
    )

    corr = X.corr().abs().fillna(0)
    distance = 1 - corr

    linkage = sch.linkage(
        squareform(distance.values, checks=False),
        method="average"
    )

    labels = sch.fcluster(
        linkage,
        t=0.2,
        criterion="distance"
    )

    cluster_map = pd.DataFrame({
        **group_values,
        "feature_id": X.columns,
        "cluster": labels
    })

    return cluster_map

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

joined_data_features.cache()
ccf_filtered.cache()
display(joined_data_features.limit(3))
display(ccf_filtered.limit(3))
DRV_GRP_COLS.append('Feature_name')
print(DRV_GRP_COLS)
print(ACT_GRP_COLS)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_f_resid = joined_data_features
df_ccf_filtered = ccf_filtered

grp_cols = DRV_GRP_COLS
act_cols = ACT_GRP_COLS

df_f_resid = df_f_resid.withColumn("feature_serie", concat_ws("__", *grp_cols)).withColumnRenamed('Date','target_date')

df_ccf_filtered = df_ccf_filtered.withColumn("feature_serie", concat_ws("__", *grp_cols))


## creating the feature_id for the feature/serie combinations that persisted after filtering
feature_series = df_ccf_filtered.select("feature_serie").distinct()
feature_series = feature_series.withColumn("feature_id", row_number().over(Window.orderBy("feature_serie")))

## ensuring there is no ambiguity in the joins

r = df_f_resid.alias("r")

initial_joined_df = feature_series.join(df_ccf_filtered, ["feature_serie"], "left").drop('Feature_name','Indicator')
joined_df = initial_joined_df.join(r, ['feature_serie',*act_cols], 'inner')
    # .drop(*[col(f"r.{c}") for c in grp_cols])\
    # .select(*grp_cols, *[col(f"r.{c}") for c in r.columns if c in act_cols if c not in grp_cols + ["feature_serie"]], "target_date", "feature_id", "feature_residual" )


schema = StructType(
    [StructField(c, StringType(), False) for c in grp_cols] +
    [
        StructField("feature_id", IntegerType(), False),
        StructField("cluster", IntegerType(), False)
    ]
)


# clustering = joined_df.select(*act_cols, *grp_cols, 'target_date','feature_id','feature_residual').groupBy(*act_cols).applyInPandas(corr_clustering, schema=schema)
# display(clustering)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df = joined_df.select(*act_cols, *grp_cols, 'target_date','feature_id','feature_residual').filter(col('Product_Category')=='ALU').toPandas()
display(df.head(1))

# grouping columns are whatever isn't a measure
group_cols = [
    c for c in df.columns
    if c not in [
        "target_date",
        "feature_id",
        "feature_residual"
    ]
]

group_cols

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def corr_clustering(df):


    group_values = {
        c: df[c].iloc[0]
        for c in group_cols
    }

    wide_df = (
        df.pivot_table(
            index=group_cols + ["target_date"],
            columns="feature_id",
            values="feature_residual",
            aggfunc="first"
        )
        .reset_index()
    )

    X = (
        wide_df
        .drop(columns=group_cols + ["target_date"])
        .select_dtypes(include=[np.number])
        .fillna(0)
    )

    corr = X.corr().abs().fillna(0)
    distance = 1 - corr

    linkage = sch.linkage(
        squareform(distance.values, checks=False),
        method="average"
    )

    labels = sch.fcluster(
        linkage,
        t=0.2,
        criterion="distance"
    )

    cluster_map = pd.DataFrame({
        **group_values,
        "feature_id": X.columns,
        "cluster": labels
    })

    return cluster_map

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************







ij = initial_joined_df.alias("ij")
c_df = clustering.alias("c")
    
conditions = []

for col_nam in act_cols:
    # Automatically map target_* -> feature_*
    conditions.append(
        col(f"ij.{col_nam}") == col(f"c.{col_nam}")
    )
# Always join on the dates
conditions.append(
    col(f"ij.feature_id") ==
    col(f"c.feature_id")
)

join_cond = reduce(operator.and_, conditions)

data_w_cluster = ij.join(
    c_df,
    join_cond, 'inner'
).select(*[col(f"ij.{c}") for c in ij.columns],c_df['cluster'])

if 'series' in grp_cols:
    cluster_corr_w = Window.partitionBy("series", "cluster").orderBy(desc("max_corr"))

    data_distinct = data_w_cluster.select("series","feature_id","cluster","max_corr").distinct()

    data_distinct = data_distinct.withColumn("cluster_rank", row_number().over(cluster_corr_w))

    data_filtered = data_distinct.filter(col("cluster_rank")<=3)

    ## joining remaining features w/ original data
    df = data_filtered.alias('df')

    post_cluster_filtering = df.join(ij, ['feature_id'], 'inner').drop(df['series'],df['max_corr'])

    ## ranking series within clusters
    ranked_w = Window. partitionBy("series").orderBy(desc('max_corr'))

    post_cluster_filtering = post_cluster_filtering.withColumn("feature_rank", dense_rank().over(ranked_w))

else:
    cluster_corr_w = Window.partitionBy("cluster").orderBy(desc("max_corr"))

    data_distinct = data_w_cluster.select("feature_id","cluster","max_corr").distinct()

    data_distinct = data_distinct.withColumn("cluster_rank", row_number().over(cluster_corr_w))

    data_filtered = data_distinct.filter(col("cluster_rank")<=3)

    ## joining remaining features w/ original data
    df = data_filtered.alias('df')

    post_cluster_filtering = df.join(ij, ['feature_id'], 'inner').drop(df['max_corr'])

    ## ranking w/o series within clusters
    ranked_w = Window.orderBy(desc('max_corr'))

    post_cluster_filtering = post_cluster_filtering.withColumn("feature_rank", dense_rank().over(ranked_w))    



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




# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def top_features(df_f_resid, ccf_filtered, DRV_GRP_COLS, ACT_GRP_COLS):
    df_f_resid = df_f_resid.withColumn("feature_serie", concat_ws("__", *grp_cols))

    df_ccf_filtered = df_ccf_filtered.withColumn("feature_serie", concat_ws("__", *grp_cols))

    ## creating the feature_id for the feature/serie combinations that persisted after filtering
    feature_series = df_ccf_filtered.select("feature_serie").distinct()
    feature_series = feature_series.withColumn("feature_id", row_number().over(Window.orderBy("feature_serie")))

    ## ensuring there is no ambiguity in the joins

    r = df_f_resid.alias("r")


    initial_joined_df = feature_series.join(df_ccf_filtered, ["feature_serie"], "left")
    joined_df = initial_joined_df.join(r, ['feature_serie'], 'inner')\
        .drop(*[col(f"r.{c}") for c in grp_cols])\
        .select(*grp_cols, "target_date", "feature_id", "feature_residual" )


    schema = StructType(
        [StructField(c, StringType(), False) for c in grp_cols] +
        [
            StructField("feature_id", IntegerType(), False),
            StructField("cluster", IntegerType(), False)
        ]
    )

    clustering = joined_df.groupBy(*act_cols).applyInPandas(corr_clustering, schema=schema)


    ij = initial_joined_df.alias("ij")
    c_df = clustering.alias("c")
        
    conditions = []

    for col_nam in act_cols:
        # Automatically map target_* -> feature_*
        conditions.append(
            col(f"ij.{col_nam}") == col(f"c.{col_nam}")
        )
    # Always join on the dates
    conditions.append(
        col(f"ij.feature_id") ==
        col(f"c.feature_id")
    )

    join_cond = reduce(operator.and_, conditions)

    data_w_cluster = ij.join(
        c_df,
        join_cond, 'inner'
    ).select(*[col(f"ij.{c}") for c in ij.columns],c_df['cluster'])

    if 'series' in grp_cols:
        cluster_corr_w = Window.partitionBy("series", "cluster").orderBy(desc("max_corr"))

        data_distinct = data_w_cluster.select("series","feature_id","cluster","max_corr").distinct()

        data_distinct = data_distinct.withColumn("cluster_rank", row_number().over(cluster_corr_w))

        data_filtered = data_distinct.filter(col("cluster_rank")<=3)

        ## joining remaining features w/ original data
        df = data_filtered.alias('df')

        post_cluster_filtering = df.join(ij, ['feature_id'], 'inner').drop(df['series'],df['max_corr'])

        ## ranking series within clusters
        ranked_w = Window. partitionBy("series").orderBy(desc('max_corr'))

        post_cluster_filtering = post_cluster_filtering.withColumn("feature_rank", dense_rank().over(ranked_w))

    else:
        cluster_corr_w = Window.partitionBy("cluster").orderBy(desc("max_corr"))

        data_distinct = data_w_cluster.select("feature_id","cluster","max_corr").distinct()

        data_distinct = data_distinct.withColumn("cluster_rank", row_number().over(cluster_corr_w))

        data_filtered = data_distinct.filter(col("cluster_rank")<=3)

        ## joining remaining features w/ original data
        df = data_filtered.alias('df')

        post_cluster_filtering = df.join(ij, ['feature_id'], 'inner').drop(df['max_corr'])

        ## ranking w/o series within clusters
        ranked_w = Window.orderBy(desc('max_corr'))

        post_cluster_filtering = post_cluster_filtering.withColumn("feature_rank", dense_rank().over(ranked_w))    



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



    return ccf_filtered

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

DRV_GRP_COLS.append('Feature_name')

filtered_features = top_features(ccf_filtered, DRV_GRP_COLS, ACT_GRP_COLS).cache()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## melt expanded features to a long format

# the 'melt_cols' are 'grouping cols' with 'Feature_name' removed 
melt_cols = DRV_GRP_COLS.copy()
if 'Feature_name' in melt_cols:
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
).cache()

print(DRV_GRP_COLS)
grp_cols = list(dict.fromkeys(DRV_GRP_COLS + ACT_GRP_COLS))

combos = filtered_features.select(*grp_cols, "Lag", "rank").distinct()

# combining the actuals original data w/ the filtered features and their identified lags
# creating the driver_date column which is the actuals data lagged by the features identified lag
target_w_features = (
    data
    .join(combos, ACT_GRP_COLS, "inner")
    .withColumn("driver_date", expr("add_months(Date, -Lag)"))
).cache()


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

results_w_col = result.withColumn('feature_col', concat_ws("__",*DRV_GRP_COLS, col("Lag").cast("string")))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Elastic Net 

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
# CONFIG for the ElasticNetCV function
# ============================================================
id_cols = ACT_GRP_COLS + ["Date"]

MIN_HISTORY = 24          # minimum months of target history required to fit
TEST_FRAC = 0.2            # holdout fraction, taken from the END (time-respecting)
N_BOOT = 100                 # bootstrap resamples for stability selection
BOOT_SAMPLE_FRAC = 0.8
STABILITY_THRESHOLD = 0.8   # keep features selected in >=60% of bootstraps
MAX_FEATURES_OUT = 10        # cap on final selected features per series


# ============================================================
# CORE FIT FUNCTION — runs once per ACT_GRP_COLS group, distributed via Spark
# ============================================================
def fit_and_select_features(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf = pdf.sort_values("Date").reset_index(drop=True)

    id_vals = {c: pdf[c].iloc[0] for c in ACT_GRP_COLS}


    wide = (
        pdf
        .pivot_table(
            index=['Date',target_col],
            columns="feature_col",
            values="Feature_value",
            aggfunc="first"
        )
        .reset_index()
    )

    # the time trend is generating a base 'feature' that is designed to 'carry' trend, allowing residuals from the target to be explained by the features
    wide['time_trend'] = np.arange(len(wide))
    


    pdf = wide.copy()

    feature_cols = [c for c in pdf.columns if c not in id_cols + [target_col]]
    




    def empty_result(reason_cols=None):
        cols = feature_cols if reason_cols is None else reason_cols
        return pd.DataFrame([{
            **id_vals, "Feature": f, "Coefficient": None, "abs_coef": None,
            "importance": None, "importance_pct": None,
            "stability_score": None, "best_alpha": None,
            "best_l1_ratio": None, "test_r2": None, "n_obs": len(pdf),
        } for f in cols])

    ###########################################################
    # Prepare X / y
    ###########################################################

    X = pdf[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(pdf[target_col], errors="coerce")

    valid = y.notna()
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)

    if len(y) < MIN_HISTORY:
        return empty_result()


    if X.shape[1] == 0:
        return empty_result(feature_cols)


    ###########################################################
    # Train/Test split (time respecting)
    ###########################################################

    # ---- time-respecting split: last TEST_FRAC as holdout, never shuffled ----
    split_idx = int(len(y) * (1 - TEST_FRAC))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(y_train) < 12 or len(y_test) < 3:
        return empty_result(feature_cols)


    ###########################################################
    # Impute
    ###########################################################

    # ---- median impute from TRAIN ONLY (no leakage) ----
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)


    ###########################################################
    # Scale
    ###########################################################

    scaler = SkScaler().fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=feature_cols)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)


    ## TIME SERIES CV
    n_splits = builtins.min(5, builtins.max(2, len(y_train) // 12))
    tscv = TimeSeriesSplit(n_splits=n_splits)


    ## ELASTIC NET CV
    try:
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
            alphas=np.logspace(-3, 1, 30),
            cv=tscv,
            max_iter=10000,
            tol=1e-6,
            random_state=42
        ).fit(X_train_s, y_train)
    except Exception:
        return empty_result(feature_cols)

    test_r2 = model.score(X_test_s, y_test)

    # ---- bootstrap stability selection on train fold ----
    stability_counts = pd.Series(0, index=feature_cols, dtype=float)
    for _ in range(N_BOOT):
        boot_idx = X_train_s.sample(frac=BOOT_SAMPLE_FRAC, replace=True).index
        try:
            m = ElasticNetCV(
                l1_ratio=[model.l1_ratio_], alphas=[model.alpha_],
                cv=2, max_iter=5000,
            ).fit(X_train_s.loc[boot_idx], y_train.loc[boot_idx])
            stability_counts += (pd.Series(m.coef_, index=feature_cols) != 0).astype(int)
        except Exception:
            continue
            
    stability_score = stability_counts / N_BOOT

    coefs = pd.Series(model.coef_, index=feature_cols)
    abs_coef = coefs.abs()
    total_abs = abs_coef.sum()
    if total_abs > 0:
        importance = abs_coef/total_abs
    else:
        importance = pd.Series(0.0, index=feature_cols)



    out = pd.DataFrame({
        **{c: id_vals[c] for c in ACT_GRP_COLS},
        "Feature": feature_cols,
        "Coefficient": coefs.values,
        "abs_coef": coefs.abs().values,
        "importance": importance.values,
        "importance_pct": (importance * 100).values,
        "stability_score": stability_score.values,
        "best_alpha": model.alpha_,
        "best_l1_ratio": model.l1_ratio_,
        "test_r2": test_r2,
        "n_obs": len(pdf),
    })



    return out



# ============================================================
# RUN — distributed across Spark executors, one fit per series
# ============================================================

# ============================================================
# OUTPUT SCHEMAS
# ============================================================
diagnostics_schema = StructType(
    [StructField(c, StringType(), False) for c in ACT_GRP_COLS] +
    [
        StructField("Feature", StringType(), False),
        StructField("Coefficient", DoubleType(), True),         # ElasticNet's fitted weight for the feature
        StructField("abs_coef", DoubleType(), True),
        StructField("importance", DoubleType(), True),
        StructField("importance_pct", DoubleType(), True),
        StructField("stability_score", DoubleType(), True),     # fraction of bootstrap resamples where the feature had a non zero coefficient
        StructField("best_alpha", DoubleType(), True),          # 0-.03 real signal, .1-1 reasonable signal to noise 3-10 little to no value add from drivers
        StructField("best_l1_ratio", DoubleType(), True),       # 0-.1 Mostly Ridge (keeps correlated features), .9-1 mostly Lasso (zeroes out redundant features)
        StructField("test_r2", DoubleType(), True),             # Negative: feature is damaging forecast, 0-.15: weak, .15-.4 good performance
        StructField("n_obs", IntegerType(), True),
    ]
)





# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

if elasticnet_run:

    results_elasticnet = results_w_col.filter(col('rank')<=elasticnet_init_features).drop('rank')

    feature_diagnostics = (
        results_elasticnet
        .filter(col(target_col).isNotNull())
        .groupBy(*ACT_GRP_COLS)
        .applyInPandas(fit_and_select_features, schema=diagnostics_schema)
    )

    feature_diagnostics.cache()

    w = (
        Window
        .partitionBy(*ACT_GRP_COLS)
        .orderBy(desc('importance'),desc('stability_score'))
    )

    selected_features = (
        feature_diagnostics
        .filter(
            (col('stability_score')>=STABILITY_THRESHOLD) &
            (col('abs_coef')!=0) &
            (col('Feature')!='time_trend') 
        )
        .withColumn('feature_rank', row_number().over(w))
        .filter(col('feature_rank')<=15)
        .select('Feature','Coefficient','importance_pct','stability_score','feature_rank')
    )


    feature_diagnostics_pdf = feature_diagnostics.toPandas()

    feature_diagnostics_pdf.to_excel(feature_diagnostics_file)


    selected_features_pdf = selected_features.toPandas()
    selected_features_pdf.to_excel(selected_feature_file)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## XGBOOST Feature Selection

# CELL ********************

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
import xgboost as xgb
import pandas as pd
import numpy as np
import builtins

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# CONFIG
# ============================================================
MIN_HISTORY = 24
TEST_FRAC = 0.2
N_BOOT = 30                    # bootstrap resamples for stability selection
BOOT_SAMPLE_FRAC = 0.8
STABILITY_THRESHOLD = 0.6
MAX_FEATURES_OUT = 25
PERM_REPEATS = 30              # repeats for permutation importance

# structural controls — absorb trend/seasonality so driver importances aren't
# penalized for failing to explain target movement they were never suited to explain
CONTROL_COLS = ["time_trend", "y_lag1", "y_lag12"]

# XGBoost hyperparameter grid — small, deliberately conservative given short series
XGB_PARAM_GRID = [
    {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 200, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 200, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 3, "learning_rate": 0.1,  "n_estimators": 100, "subsample": 0.7, "colsample_bytree": 0.7},
    {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 150, "subsample": 0.8, "colsample_bytree": 0.6},
]


# ============================================================
# OUTPUT SCHEMA
# ============================================================
xgb_diagnostics_schema = StructType(
    [StructField(c, StringType(), False) for c in ACT_GRP_COLS] +
    [
        StructField("Feature", StringType(), False),
        StructField("importance_gain", DoubleType(), True),      # split-quality based
        StructField("importance_perm", DoubleType(), True),      # permutation based (primary)
        StructField("importance_perm_std", DoubleType(), True),
        StructField("importance_pct", DoubleType(), True),
        StructField("stability_score", DoubleType(), True),
        StructField("selected", IntegerType(), True),
        StructField("is_control", IntegerType(), False),
        StructField("best_max_depth", IntegerType(), True),
        StructField("best_learning_rate", DoubleType(), True),
        StructField("best_n_estimators", IntegerType(), True),
        StructField("test_r2", DoubleType(), True),
        StructField("controls_only_r2", DoubleType(), True),
        StructField("naive_r2", DoubleType(), True),
        StructField("drivers_add_value", IntegerType(), True),
        StructField("n_obs", IntegerType(), True),
    ]
)

# ============================================================
# CORE FIT FUNCTION — one XGBoost model per ACT_GRP_COLS group
# ============================================================
def fit_xgb_feature_importance(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf = pdf.sort_values("Date").reset_index(drop=True)
    id_vals = {c: pdf[c].iloc[0] for c in ACT_GRP_COLS}

    # ---- pivot long -> wide, one row per Date ----
    wide = (
        pdf.pivot_table(index=["Date", target_col], columns="feature_col",
                         values="Feature_value", aggfunc="first")
        .reset_index()
    )

    driver_cols = [c for c in wide.columns if c not in ["Date", target_col]]

    # ---- structural controls ----
    wide["time_trend"] = np.arange(len(wide))
    wide["y_lag1"] = wide[target_col].shift(1)
    wide["y_lag12"] = wide[target_col].shift(12)
    wide = wide.dropna(subset=["y_lag12"]).reset_index(drop=True)


    feature_cols = driver_cols + CONTROL_COLS

    def empty_result(cols=None, n_obs=0):
        cols = feature_cols if cols is None else cols
        return pd.DataFrame([{
            **id_vals, "Feature": f,
            "importance_gain": None, "importance_perm": None, "importance_perm_std": None,
            "importance_pct": None, "stability_score": None, "selected": 0,
            "is_control": int(f in CONTROL_COLS),
            "best_max_depth": None, "best_learning_rate": None, "best_n_estimators": None,
            "test_r2": None, "controls_only_r2": None, "naive_r2": None,
            "drivers_add_value": None, "n_obs": n_obs,
        } for f in cols])

    if len(wide) < MIN_HISTORY or len(driver_cols) == 0:
        return empty_result(feature_cols, len(wide))

    X = wide[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(wide[target_col], errors="coerce")

    valid = y.notna()
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)

    if len(y) < MIN_HISTORY:
        return empty_result(feature_cols, len(wide))

    # ---- time-respecting split ----
    split_idx = int(len(y) * (1 - TEST_FRAC))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(y_train) < 12 or len(y_test) < 3:
        return empty_result(feature_cols, len(wide))

    # ---- median impute from TRAIN only (no leakage) ----
    medians = X_train.median()
    #medians = medians.fillna(0)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    # ---- hyperparameter search via TimeSeriesSplit CV ----
    n_splits = builtins.min(5, builtins.max(2, len(y_train) // 12))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    best_score, best_params = -np.inf, None
    for params in XGB_PARAM_GRID:
        fold_scores = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            try:
                m = xgb.XGBRegressor(
                    objective="reg:squarederror", random_state=42,
                    n_jobs=1,  # avoid nested parallelism fighting Spark's own executor threads
                    **params
                ).fit(X_tr, y_tr)
                fold_scores.append(m.score(X_val, y_val))
            except Exception:
                fold_scores.append(-np.inf)
        mean_score = np.mean(fold_scores)
        if mean_score > best_score:
            best_score, best_params = mean_score, params

    if best_params is None:
        return empty_result(feature_cols, len(wide))

    # ---- fit final model on full training set with best params ----
    try:
        model = xgb.XGBRegressor(
            objective="reg:squarederror", random_state=42, n_jobs=1, **best_params
        ).fit(X_train, y_train)
    except Exception:
        return empty_result(feature_cols, len(wide))

    test_r2 = model.score(X_test, y_test)
    naive_r2 = r2_score(y_test, np.full_like(y_test, y_train.mean(), dtype=float))

    # ---- controls-only baseline: isolates drivers' marginal contribution ----
    ctrl_model = xgb.XGBRegressor(
        objective="reg:squarederror", random_state=42, n_jobs=1, **best_params
    ).fit(X_train[CONTROL_COLS], y_train)
    controls_only_r2 = ctrl_model.score(X_test[CONTROL_COLS], y_test)
    drivers_add_value = int(test_r2 > controls_only_r2)

    # ---- gain-based importance (fast, built-in, but biased toward high-cardinality splits) ----
    gain_importance = pd.Series(model.feature_importances_, index=feature_cols)

    # ---- permutation importance on TEST set (primary metric — unbiased, model-agnostic) ----
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=PERM_REPEATS, random_state=42, n_jobs=1
    )
    perm_importance = pd.Series(perm.importances_mean, index=feature_cols).clip(lower=0)
    perm_importance_std = pd.Series(perm.importances_std, index=feature_cols)

    total_perm = perm_importance.sum()
    importance_pct = (perm_importance / total_perm * 100) if total_perm > 0 else perm_importance * 0

    # ---- bootstrap stability selection ----
    stability_counts = pd.Series(0, index=feature_cols, dtype=float)
    for _ in range(N_BOOT):
        boot_idx = X_train.sample(frac=BOOT_SAMPLE_FRAC, replace=True).index
        try:
            m = xgb.XGBRegressor(
                objective="reg:squarederror", random_state=42, n_jobs=1, **best_params
            ).fit(X_train.loc[boot_idx], y_train.loc[boot_idx])
            boot_perm = permutation_importance(
                m, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1
            )
            # a feature "counts" this round if its permutation importance is meaningfully > 0
            stability_counts += (pd.Series(boot_perm.importances_mean, index=feature_cols) > 1e-6).astype(int)
        except Exception:
            continue
    stability_score = stability_counts / N_BOOT

    selected = (
        (stability_score >= STABILITY_THRESHOLD) &
        (perm_importance > 0) &
        (~pd.Series(feature_cols, index=feature_cols).isin(CONTROL_COLS))  # controls never "selected"
        #&   drivers_add_value  # series-level gate: only trust selection if drivers beat controls-only
    )

    out = pd.DataFrame({
        **{c: id_vals[c] for c in ACT_GRP_COLS},
        "Feature": feature_cols,
        "importance_gain": gain_importance.values,
        "importance_perm": perm_importance.values,
        "importance_perm_std": perm_importance_std.values,
        "importance_pct": importance_pct.values,
        "stability_score": stability_score.values,
        "selected": selected.astype(int).values,
        "is_control": [int(f in CONTROL_COLS) for f in feature_cols],
        "best_max_depth": best_params["max_depth"],
        "best_learning_rate": best_params["learning_rate"],
        "best_n_estimators": best_params["n_estimators"],
        "test_r2": test_r2,
        "controls_only_r2": controls_only_r2,
        "naive_r2": naive_r2,
        "drivers_add_value": int(drivers_add_value),
        "n_obs": len(wide),
    })

    # cap selected drivers at MAX_FEATURES_OUT, ranked by importance_pct
    out = out.sort_values(["selected", "importance_pct"], ascending=[False, False])
    keep_mask = out["selected"] == 1
    if keep_mask.sum() > MAX_FEATURES_OUT:
        drop_idx = out[keep_mask].index[MAX_FEATURES_OUT:]
        out.loc[drop_idx, "selected"] = 0

    return out



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

if xgboost_run:
    # ============================================================
    # RUN — distributed across Spark executors, one fit per series
    # ============================================================

    results_xgb = results_w_col.filter(col('rank')<=xgboost_init_features).drop('rank')


    xgb_feature_diagnostics = (
        results_xgb
        .filter(col(target_col).isNotNull())
        .groupBy(*ACT_GRP_COLS)
        .applyInPandas(fit_xgb_feature_importance, schema=xgb_diagnostics_schema)
    )

    xgb_feature_diagnostics.cache()
    display(xgb_feature_diagnostics.orderBy(*ACT_GRP_COLS, desc("importance_pct")))

    xgb_selected_features = xgb_feature_diagnostics.filter(col('selected')==1).orderBy(*ACT_GRP_COLS,desc('importance_pct'))

    xgb_feature_diagnostics_pdf = xgb_feature_diagnostics.toPandas()

    xgb_feature_diagnostics_pdf.to_excel(xgb_feature_diagnostics_file)


    xgb_selected_features_pdf = xgb_selected_features.toPandas()
    xgb_selected_features_pdf.to_excel(xgb_selected_feature_file)


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
