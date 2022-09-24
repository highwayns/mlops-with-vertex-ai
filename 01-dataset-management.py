#!/usr/bin/env python
# coding: utf-8

# # 01 - Data Analysis and Preparation
# 
# This notebook covers the following tasks:
# 
# 1. Perform exploratory data analysis and visualization.
# 2. Prepare the data for the ML task in BigQuery.
# 3. Generate and fix a ` TFDV schema` for the source data.
# 4. Create a `Vertex Dataset resource` dataset.
# 

# ## Dataset
# 
# The [Chicago Taxi Trips](https://pantheon.corp.google.com/marketplace/details/city-of-chicago-public-data/chicago-taxi-trips) dataset is one of [public datasets hosted with BigQuery](https://cloud.google.com/bigquery/public-data/), which includes taxi trips from 2013 to the present, reported to the City of Chicago in its role as a regulatory agency. The `taxi_trips` table size is 70.72 GB and includes more than 195 million records. The dataset includes information about the trips, like pickup and dropoff datetime and location, passengers count, miles travelled, and trip toll. 
# 
# The ML task is to predict whether a given trip will result in a tip > 20%.

# ## Setup

# ### Import libraries

# In[ ]:


import os
import pandas as pd
import tensorflow as tf
import tensorflow_data_validation as tfdv
from google.cloud import bigquery
import matplotlib.pyplot as plt

from google.cloud import aiplatform as vertex_ai


# ### Setup Google Cloud project

# In[ ]:


PROJECT = '[your-project-id]' # Change to your project id.
REGION = 'us-central1' # Change to your region.

if PROJECT == "" or PROJECT is None or PROJECT == "[your-project-id]":
    # Get your GCP project id from gcloud
    shell_output = get_ipython().getoutput("gcloud config list --format 'value(core.project)' 2>/dev/null")
    PROJECT = shell_output[0]
    
print("Project ID:", PROJECT)
print("Region:", REGION)


# ### Set configurations

# In[ ]:


BQ_DATASET_NAME = 'playground_us' # Change to your BQ dataset name.
BQ_TABLE_NAME = 'chicago_taxitrips_prep'
BQ_LOCATION = 'US'

DATASET_DISPLAY_NAME = 'chicago-taxi-tips'

RAW_SCHEMA_DIR = 'src/raw_schema'


# ## 1. Explore the data in BigQuery

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'data', "\nSELECT \n    CAST(EXTRACT(DAYOFWEEK FROM trip_start_timestamp) AS string) AS trip_dayofweek, \n    FORMAT_DATE('%A',cast(trip_start_timestamp as date)) AS trip_dayname,\n    COUNT(*) as trip_count,\nFROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`\nWHERE\n    EXTRACT(YEAR FROM trip_start_timestamp) = 2015 \nGROUP BY\n    trip_dayofweek,\n    trip_dayname\nORDER BY\n    trip_dayofweek\n;")


# In[ ]:


data


# In[ ]:


data.plot(kind='bar', x='trip_dayname', y='trip_count')


# ## 2. Create data for the ML task
# 
# We add a `ML_use` column for pre-splitting the data, where 80% of the datsa items are set to `UNASSIGNED` while the other 20% is set to `TEST`.
# 
# This column is used during training to split the dataset for training and test.
# 
# In the training phase, the `UNASSIGNED` are split into `train` and `eval`. The `TEST` split is will be used for the final model validation.

# ### Create destination BigQuery dataset

# In[ ]:


get_ipython().system('bq --location=$BQ_LOCATION mk -d $PROJECT:$BQ_DATASET_NAME')


# In[ ]:


sample_size = 1000000
year = 2020


# In[ ]:


sql_script = '''
CREATE OR REPLACE TABLE `@PROJECT.@DATASET.@TABLE` 
AS (
    WITH
      taxitrips AS (
      SELECT
        trip_start_timestamp,
        trip_seconds,
        trip_miles,
        payment_type,
        pickup_longitude,
        pickup_latitude,
        dropoff_longitude,
        dropoff_latitude,
        tips,
        fare
      FROM
        `bigquery-public-data.chicago_taxi_trips.taxi_trips`
      WHERE 1=1 
      AND pickup_longitude IS NOT NULL
      AND pickup_latitude IS NOT NULL
      AND dropoff_longitude IS NOT NULL
      AND dropoff_latitude IS NOT NULL
      AND trip_miles > 0
      AND trip_seconds > 0
      AND fare > 0
      AND EXTRACT(YEAR FROM trip_start_timestamp) = @YEAR
    )

    SELECT
      trip_start_timestamp,
      EXTRACT(MONTH from trip_start_timestamp) as trip_month,
      EXTRACT(DAY from trip_start_timestamp) as trip_day,
      EXTRACT(DAYOFWEEK from trip_start_timestamp) as trip_day_of_week,
      EXTRACT(HOUR from trip_start_timestamp) as trip_hour,
      trip_seconds,
      trip_miles,
      payment_type,
      ST_AsText(
          ST_SnapToGrid(ST_GeogPoint(pickup_longitude, pickup_latitude), 0.1)
      ) AS pickup_grid,
      ST_AsText(
          ST_SnapToGrid(ST_GeogPoint(dropoff_longitude, dropoff_latitude), 0.1)
      ) AS dropoff_grid,
      ST_Distance(
          ST_GeogPoint(pickup_longitude, pickup_latitude), 
          ST_GeogPoint(dropoff_longitude, dropoff_latitude)
      ) AS euclidean,
      CONCAT(
          ST_AsText(ST_SnapToGrid(ST_GeogPoint(pickup_longitude,
              pickup_latitude), 0.1)), 
          ST_AsText(ST_SnapToGrid(ST_GeogPoint(dropoff_longitude,
              dropoff_latitude), 0.1))
      ) AS loc_cross,
      IF((tips/fare >= 0.2), 1, 0) AS tip_bin,
      IF(RAND() <= 0.8, 'UNASSIGNED', 'TEST') AS ML_use
    FROM
      taxitrips
    LIMIT @LIMIT
)
'''


# In[ ]:


sql_script = sql_script.replace(
    '@PROJECT', PROJECT).replace(
    '@DATASET', BQ_DATASET_NAME).replace(
    '@TABLE', BQ_TABLE_NAME).replace(
    '@YEAR', str(year)).replace(
    '@LIMIT', str(sample_size))


# In[ ]:


print(sql_script)


# In[ ]:


bq_client = bigquery.Client(project=PROJECT, location=BQ_LOCATION)
job = bq_client.query(sql_script)
_ = job.result()


# In[ ]:


get_ipython().run_cell_magic('bigquery', '--project {PROJECT}', '\nSELECT ML_use, COUNT(*)\nFROM playground_us.chicago_taxitrips_prep # Change to your BQ dataset and table names.\nGROUP BY ML_use')


# ### Load a sample data to a Pandas DataFrame

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'sample_data --project {PROJECT}', '\nSELECT * EXCEPT (trip_start_timestamp, ML_use)\nFROM playground_us.chicago_taxitrips_prep # Change to your BQ dataset and table names.')


# In[ ]:


sample_data.head().T


# In[ ]:


sample_data.tip_bin.value_counts()


# In[ ]:


sample_data.euclidean.hist()


# ## 3. Generate raw data schema
# 
# The [TensorFlow Data Validation (TFDV)](https://www.tensorflow.org/tfx/data_validation/get_started) data schema will be used in:
# 1. Identify the raw data types and shapes in the data transformation.
# 2. Create the serving input signature for the custom model.
# 3. Validate the new raw training data in the TFX pipeline.

# In[ ]:


stats = tfdv.generate_statistics_from_dataframe(
    dataframe=sample_data,
    stats_options=tfdv.StatsOptions(
        label_feature='tip_bin',
        weight_feature=None,
        sample_rate=1,
        num_top_values=50
    )
)


# In[ ]:


tfdv.visualize_statistics(stats)


# In[ ]:


schema = tfdv.infer_schema(statistics=stats)
tfdv.display_schema(schema=schema)


# In[ ]:


raw_schema_location = os.path.join(RAW_SCHEMA_DIR, 'schema.pbtxt')
tfdv.write_schema_text(schema, raw_schema_location)


# ## 4. Create Vertex Dataset resource

# In[ ]:


vertex_ai.init(
    project=PROJECT,
    location=REGION
)


# ### Create the dataset resource

# In[ ]:


bq_uri = f"bq://{PROJECT}.{BQ_DATASET_NAME}.{BQ_TABLE_NAME}"

dataset = vertex_ai.TabularDataset.create(
    display_name=DATASET_DISPLAY_NAME, bq_source=bq_uri)

dataset.gca_resource


# ### Get the dataset resource
# 
# The dataset resource is retrieved by display name. Because multiple datasets can have the same display name, we retrieve the most recent updated one.

# In[ ]:


dataset = vertex_ai.TabularDataset.list(
    filter=f"display_name={DATASET_DISPLAY_NAME}", 
    order_by="update_time")[-1]

print("Dataset resource name:", dataset.resource_name)
print("Dataset BigQuery source:", dataset.gca_resource.metadata['inputConfig']['bigquerySource']['uri'])


# In[ ]:




