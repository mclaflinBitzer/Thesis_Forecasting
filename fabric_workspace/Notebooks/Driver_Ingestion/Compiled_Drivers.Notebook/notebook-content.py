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
# META     },
# META     "warehouse": {
# META       "known_warehouses": []
# META     }
# META   }
# META }

# CELL ********************

from pyspark.sql.functions import *
from itertools import chain
from pyspark.sql.window import Window

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Extracting all tables within the lakehouse

# CELL ********************

lakehouse = "Sales_Forecasting.bronze"
tables = spark.catalog.listTables(lakehouse)
tables

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Extracting Driver tables and unioning into one dataframe

# CELL ********************

driver_tables= []
for t in tables:
    if t.name.startswith("drv"):
        print(f"{t.name} is a driver table & has been added to the driver_table list")
        driver_tables.append(t.name)
    else:
        print(f"{t.name} is not a driver table and has been skipped")


dfs = []
print("reading tables")

for name in driver_tables:
    print(f"Reading {name}")
    df = spark.read.table(lakehouse+"."+name).select("Country", "Indicator", "Date", "Value")
    df = df.withColumn("source_table", lit(name))
    dfs.append(df)

if dfs:
    joined_df = dfs[0]
    for df in dfs[1:]:
        joined_df = joined_df.unionByName(df, allowMissingColumns=True)
else:
    joined_df = None


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(joined_df.limit(5))
display(joined_df.select("source_table").distinct())
display(joined_df.count())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Read and apply Region Mapping 

# CELL ********************

country_region_map = {

    # --------------------
    # Special
    # --------------------
    "World": "World",
    "Grand Total": "World",

    # --------------------
    # China
    # --------------------
    "China": "China",
    "Hong Kong": "China",
    "Hong Kong (China SAR)": "China",
    "Hong Kong,  China": "China",
    "Macau (SAR)": "China",
    "Macao": "China",
    "Macao (China SAR)": "China",
    "Taiwan": "China",
    "Taiwan (Province of China)": "China",

    # --------------------
    # North America
    # --------------------
    "United States": "N.America",
    "Canada": "N.America",
    "Mexico": "N.America",
    "United States of America": "N.America",
    "Puerto Rico": "N.America",
    "American Samoa": "N.America",
    "Guam": "N.America",
    "Northern Mariana Islands": "N.America",
    "U.S. Virgin Islands": "N.America",
    "British Virgin Islands": "N.America",
    "Turks and Caicos Islands": "N.America",
    "Cayman Islands": "N.America",
    "Anguilla": "N.America",
    "Bahamas": "N.America",
    "Barbados": "N.America",
    "Bermuda": "N.America",
    "Cuba": "N.America",
    "Curacao": "N.America",
    "Dominica": "N.America",
    "Dominican Republic": "N.America",
    "Grenada": "N.America",
    "Guatemala": "N.America",
    "Haiti": "N.America",
    "Honduras": "N.America",
    "Jamaica": "N.America",
    "Saint Kitts and Nevis": "N.America",
    "Saint Lucia": "N.America",
    "Saint Martin": "N.America",
    "Saint Pierre and Miquelon": "N.America",
    "Saint Vincent and the Grenadines": "N.America",
    "Sint Maarten": "N.America",
    "Trinidad and Tobago": "N.America",
    "Belize": "N.America",
    "Costa Rica": "N.America",
    "El Salvador": "N.America",
    "Nicaragua": "N.America",
    "Panama": "N.America",
    "Antigua and Barbuda": "N.America",
    "Montserrat": "N.America",

    # --------------------
    # South America
    # --------------------
    "Argentina": "S.America",
    "Bolivia": "S.America",
    "Brazil": "S.America",
    "Chile": "S.America",
    "Colombia": "S.America",
    "Ecuador": "S.America",
    "French Guiana": "S.America",
    "Guyana": "S.America",
    "Paraguay": "S.America",
    "Peru": "S.America",
    "Suriname": "S.America",
    "Uruguay": "S.America",
    "Venezuela": "S.America",
    "Venezuela,  RB": "S.America",
    "Aruba": "S.America",

    # --------------------
    # APAC
    # --------------------
    "Afghanistan": "APAC",
    "Bangladesh": "APAC",
    "Bhutan": "APAC",
    "Brunei": "APAC",
    "Cambodia": "APAC",
    "East Timor": "APAC",
    "Timor-Leste": "APAC",
    "Fiji": "APAC",
    "Indonesia": "APAC",
    "Japan": "APAC",
    "Kazakhstan": "APAC",
    "Kiribati": "APAC",
    "Kyrgyzstan": "APAC",
    "Laos": "APAC",
    "Maldives": "APAC",
    "Marshall Islands": "APAC",
    "Micronesia": "APAC",
    "Mongolia": "APAC",
    "Myanmar": "APAC",
    "Nepal": "APAC",
    "New Zealand": "APAC",
    "North Korea": "APAC",
    "Palau": "APAC",
    "Papua New Guinea": "APAC",
    "Philippines": "APAC",
    "Samoa": "APAC",
    "Solomon Islands": "APAC",
    "South Korea": "APAC",
    "Sri Lanka": "APAC",
    "Thailand": "APAC",
    "Tajikistan": "APAC",
    "Tonga": "APAC",
    "Turkmenistan": "APAC",
    "Tuvalu": "APAC",
    "Uzbekistan": "APAC",
    "Vanuatu": "APAC",
    "Vietnam": "APAC",
    "Malaysia": "APAC",
    "Singapore": "APAC",
    "India": "APAC",
    "Australia": "APAC",
    "Pakistan": "APAC",
    "New Caledonia": "APAC",
    "Nauru": "APAC",
    "French Polynesia": "APAC",
    "Cook Islands": "APAC",
    "Wallis and Futuna": "APAC",

    # --------------------
    # EMEA
    # --------------------
    "Russia": "EMEA",
    "Germany": "EMEA",
    "Sweden": "EMEA",
    "Iraq": "EMEA",
    "Greece": "EMEA",
    "Algeria": "EMEA",
    "Slovakia": "EMEA",
    "Slovak Republic": "EMEA",
    "Angola": "EMEA",
    "Belgium": "EMEA",
    "Qatar": "EMEA",
    "Finland": "EMEA",
    "Ghana": "EMEA",
    "Belarus": "EMEA",
    "Kuwait": "EMEA",
    "Croatia": "EMEA",
    "Nigeria": "EMEA",
    "Lithuania": "EMEA",
    "Norway": "EMEA",
    "Spain": "EMEA",
    "Czechia": "EMEA",
    "Czech Republic": "EMEA",
    "Denmark": "EMEA",
    "Iran": "EMEA",
    "Iran, Islamic Rep.": "EMEA",
    "Ireland": "EMEA",
    "Morocco": "EMEA",
    "Ukraine": "EMEA",
    "Israel": "EMEA",
    "Oman": "EMEA",
    "Estonia": "EMEA",
    "Azerbaijan": "EMEA",
    "Tunisia": "EMEA",
    "Saudi Arabia": "EMEA",
    "Switzerland": "EMEA",
    "Zambia": "EMEA",
    "Ethiopia": "EMEA",
    "Latvia": "EMEA",
    "Turkey": "EMEA",
    "Turkiye": "EMEA",
    "United Arab Emirates": "EMEA",
    "Kenya": "EMEA",
    "Slovenia": "EMEA",
    "Tanzania": "EMEA",
    "Poland": "EMEA",
    "Cameroon": "EMEA",
    "Romania": "EMEA",
    "Bulgaria": "EMEA",
    "Egypt": "EMEA",
    "Egypt, Arab Rep.": "EMEA",
    "Bahrain": "EMEA",
    "Hungary": "EMEA",
    "United Kingdom": "EMEA",
    "Netherlands": "EMEA",
    "Chad": "EMEA",
    "Senegal": "EMEA",
    "Djibouti": "EMEA",
    "Malawi": "EMEA",
    "Ivory Coast": "EMEA",
    "Jordan": "EMEA",
    "Rwanda": "EMEA",
    "Sudan": "EMEA",
    "Kosovo": "EMEA",
    "Equatorial Guinea": "EMEA",
    "Albania": "EMEA",
    "Benin": "EMEA",
    "Malta": "EMEA",
    "Palestinian Territory": "EMEA",
    "Central African Republic": "EMEA",
    "Swaziland": "EMEA",
    "Eswatini": "EMEA",
    "Democratic Republic of the Congo": "EMEA",
    "Republic of the Congo": "EMEA",
    "Monaco": "EMEA",
    "Cape Verde": "EMEA",
    "Iceland": "EMEA",
    "Cyprus": "EMEA",
    "Gibraltar": "EMEA",
    "Georgia": "EMEA",
    "Montenegro": "EMEA",
    "Libya": "EMEA",
    "Armenia": "EMEA",
    "Syria": "EMEA",
    "Uganda": "EMEA",
    "South Sudan": "EMEA",
    "Lebanon": "EMEA",
    "Luxembourg": "EMEA",
    "Bosnia and Herzegovina": "EMEA",
    "Serbia": "EMEA",
    "Burkina Faso": "EMEA",
    "Mauritius": "EMEA",
    "Moldova": "EMEA",
    "Yemen": "EMEA",
    "Eritrea": "EMEA",
    "Comoros": "EMEA",
    "Saint Helena": "EMEA",
    "Togo": "EMEA",
    "San Marino": "EMEA",
    "Lesotho": "EMEA",
    "Madagascar": "EMEA",
    "Sierra Leone": "EMEA",
    "Sao Tome and Principe": "EMEA",
    "Somalia": "EMEA",
    "Burundi": "EMEA",
    "Andorra": "EMEA",
    "Gabon": "EMEA",
    "Mauritania": "EMEA",
    "Niger": "EMEA",
    "Liechtenstein": "EMEA",
    "Guinea": "EMEA",
    "Guinea-Bissau": "EMEA",
    "Seychelles": "EMEA",
    "North Macedonia": "EMEA",
    "Macedonia": "EMEA",
    "Gambia": "EMEA",
    "Botswana": "EMEA",
    "Greenland": "EMEA",
    "Liberia": "EMEA",
    "Mali": "EMEA",
    "Western Sahara": "EMEA",
    "Faroe Islands": "EMEA",
    "France": "EMEA",
    "Italy": "EMEA",
    "Portugal": "EMEA",
    "Austria": "EMEA",
    "South Africa": "EMEA",
    "Mozambique": "EMEA",
    "Namibia": "EMEA",
    "Zimbabwe": "EMEA",
}

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

mapping_expr = create_map(
    [lit(x) for x in chain(*country_region_map.items())]
)

df_mapping = joined_df.withColumn(
    "Region",
    mapping_expr[col("Country")]
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_mapping.write.format("delta").mode("overwrite").saveAsTable("Sales_Forecasting.silver.Compiled_Drivers")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
