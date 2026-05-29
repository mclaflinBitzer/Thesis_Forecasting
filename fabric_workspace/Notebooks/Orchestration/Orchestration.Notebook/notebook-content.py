# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {}
# META }

# CELL ********************

import requests

workspace_id = "991f5e4b-c174-4ff2-992e-feb17d49d25a"

token = mssparkutils.credentials.getToken("https://api.fabric.microsoft.com")

headers = {
    "Authorization": f"Bearer {token}"
}

url = f"https://api.fabric.microsoft.com/v1/workspaces/{workspace_id}/items"

response = requests.get(url, headers=headers)
data = response.json()
items = data["value"]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## filtering for the driver notebooks

drv_notebooks = [
    {
        "id": i["id"],
        "name": i["displayName"]
    }
    for i in items
    if i["type"] == "Notebook"
    and i["displayName"].startswith("DRV")
]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## export the list of notebooks to the pipeline
mssparkutils.notebook.exit(drv_notebooks)

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
