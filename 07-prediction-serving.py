#!/usr/bin/env python
# coding: utf-8

# # 07 - Prediction Serving
# 
# The purpose of the notebook is to show how to use the deployed model for online and batch prediction.
# The notebook covers the following tasks:
# 1. Test the endpoints for online prediction.
# 2. Use the uploaded custom model for batch prediction.
# 3. Run a the batch prediction pipeline using `Vertex Pipelines`.

# ## Setup

# ### Import libraries

# In[ ]:


import os
from datetime import datetime
import tensorflow as tf

from google.cloud import aiplatform as vertex_ai


# ### Setup Google Cloud project

# In[ ]:


PROJECT = '[your-project-id]' # Change to your project id.
REGION = 'us-central1' # Change to your region.
BUCKET = '[your-bucket-name]' # Change to your bucket name.

if PROJECT == "" or PROJECT is None or PROJECT == "[your-project-id]":
    # Get your GCP project id from gcloud
    shell_output = get_ipython().getoutput("gcloud config list --format 'value(core.project)' 2>/dev/null")
    PROJECT = shell_output[0]
    
if BUCKET == "" or BUCKET is None or BUCKET == "[your-bucket-name]":
    # Get your bucket name to GCP project id
    BUCKET = PROJECT
    # Try to create the bucket if it doesn't exists
    get_ipython().system(' gsutil mb -l $REGION gs://$BUCKET')
    print("")
    
print("Project ID:", PROJECT)
print("Region:", REGION)
print("Bucket name:", BUCKET)


# ### Set configurations

# In[ ]:


VERSION = 'v01'
DATASET_DISPLAY_NAME = 'chicago-taxi-tips'
MODEL_DISPLAY_NAME = f'{DATASET_DISPLAY_NAME}-classifier-{VERSION}'
ENDPOINT_DISPLAY_NAME = f'{DATASET_DISPLAY_NAME}-classifier'

SERVE_BQ_DATASET_NAME = 'playground_us' # Change to your serving BigQuery dataset name.
SERVE_BQ_TABLE_NAME = 'chicago_taxitrips_prep' # Change to your serving BigQuery table name.


# ## 1. Making Online Predicitons
# 

# In[ ]:


vertex_ai.init(
    project=PROJECT,
    location=REGION,
    staging_bucket=BUCKET
)

endpoint_name = vertex_ai.Endpoint.list(
    filter=f'display_name={ENDPOINT_DISPLAY_NAME}', 
    order_by="update_time")[-1].gca_resource.name

endpoint = vertex_ai.Endpoint(endpoint_name)


# In[ ]:


test_instances = [  
    {
        "dropoff_grid": ["POINT(-87.6 41.9)"],
        "euclidean": [2064.2696],
        "loc_cross": [""],
        "payment_type": ["Credit Card"],
        "pickup_grid": ["POINT(-87.6 41.9)"],
        "trip_miles": [1.37],
        "trip_day": [12],
        "trip_hour": [16],
        "trip_month": [2],
        "trip_day_of_week": [4],
        "trip_seconds": [555]
    }
]


# In[ ]:


predictions = endpoint.predict(test_instances).predictions

for prediction in predictions:
    print(prediction)


# In[ ]:


explanations = endpoint.explain(test_instances).explanations

for explanation in explanations:
    print(explanation)


# ## 2. Batch Prediction

# In[ ]:


WORKSPACE = f"gs://{BUCKET}/{DATASET_DISPLAY_NAME}/"
SERVING_DATA_DIR = os.path.join(WORKSPACE, 'serving_data')
SERVING_INPUT_DATA_DIR = os.path.join(SERVING_DATA_DIR, 'input_data')
SERVING_OUTPUT_DATA_DIR = os.path.join(SERVING_DATA_DIR, 'output_predictions')


# In[ ]:


if tf.io.gfile.exists(SERVING_DATA_DIR):
    print("Removing previous serving data...")
    tf.io.gfile.rmtree(SERVING_DATA_DIR)
    
print("Creating serving data directory...")
tf.io.gfile.mkdir(SERVING_DATA_DIR)
print("Serving data directory is ready.")


# ### Extract serving data to Cloud Storage as JSONL

# In[ ]:


from src.common import datasource_utils
from src.preprocessing import etl


# In[ ]:


LIMIT = 10000

sql_query = datasource_utils.get_serving_source_query(
    bq_dataset_name=SERVE_BQ_DATASET_NAME, 
    bq_table_name=SERVE_BQ_TABLE_NAME,
    limit=LIMIT
)

print(sql_query)


# In[ ]:


job_name = f"extract-{DATASET_DISPLAY_NAME}-serving-{datetime.now().strftime('%Y%m%d%H%M%S')}"

args = {
    'job_name': job_name,
    #'runner': 'DataflowRunner',
    'sql_query': sql_query,
    'exported_data_prefix': os.path.join(SERVING_INPUT_DATA_DIR, "data-"),
    'temporary_dir': os.path.join(WORKSPACE, 'tmp'),
    'gcs_location': os.path.join(WORKSPACE, 'bq_tmp'),
    'project': PROJECT,
    'region': REGION,
    'setup_file': './setup.py'
}


# In[ ]:


tf.get_logger().setLevel('ERROR')

print("Data extraction started...")
etl.run_extract_pipeline(args)
print("Data extraction completed.")


# In[ ]:


get_ipython().system('gsutil ls {SERVING_INPUT_DATA_DIR}')


# ### Submit the batch prediction job

# In[ ]:


model_name =  vertex_ai.Model.list(
    filter=f'display_name={MODEL_DISPLAY_NAME}',
    order_by="update_time")[-1].gca_resource.name


# In[ ]:


job_resources =  {
    "machine_type": 'n1-standard-2',
    #'accelerator_count': 1,
    #'accelerator_type': 'NVIDIA_TESLA_T4'
    "starting_replica_count": 1,
    "max_replica_count": 10,
}

job_display_name = f"{MODEL_DISPLAY_NAME}-prediction-job-{datetime.now().strftime('%Y%m%d%H%M%S')}"

vertex_ai.BatchPredictionJob.create(
    job_display_name=job_display_name,
    model_name=model_name,
    gcs_source=SERVING_INPUT_DATA_DIR + '/*.jsonl',
    gcs_destination_prefix=SERVING_OUTPUT_DATA_DIR,
    instances_format='jsonl',
    predictions_format='jsonl',
    sync=True,
    **job_resources,
)


# ## 3. Run the batch prediction pipeline using Vertex Pipelines

# In[ ]:


WORKSPACE = f"gs://{BUCKET}/{DATASET_DISPLAY_NAME}/"
ARTIFACT_STORE = os.path.join(WORKSPACE, 'tfx_artifacts')
PIPELINE_NAME = f'{MODEL_DISPLAY_NAME}-predict-pipeline'


# ### Set the pipeline configurations for the Vertex AI run

# In[ ]:


os.environ["PROJECT"] = PROJECT
os.environ["REGION"] = REGION
os.environ["GCS_LOCATION"] = f"gs://{BUCKET}/{DATASET_DISPLAY_NAME}"
os.environ["MODEL_DISPLAY_NAME"] = MODEL_DISPLAY_NAME
os.environ["PIPELINE_NAME"] = PIPELINE_NAME
os.environ["ARTIFACT_STORE_URI"] = ARTIFACT_STORE
os.environ["BATCH_PREDICTION_BQ_DATASET_NAME"] = SERVE_BQ_DATASET_NAME
os.environ["BATCH_PREDICTION_BQ_TABLE_NAME"] = SERVE_BQ_TABLE_NAME
os.environ["SERVE_LIMIT"] = "1000"
os.environ["BEAM_RUNNER"] = "DirectRunner"
os.environ["TFX_IMAGE_URI"] = f"gcr.io/{PROJECT}/{DATASET_DISPLAY_NAME}:{VERSION}"


# In[ ]:


import importlib
from src.tfx_pipelines import config
importlib.reload(config)

for key, value in config.__dict__.items():
    if key.isupper(): print(f'{key}: {value}')


# ### (Optional) Build the ML container image
# 
# This is the `TFX` runtime environment for the training pipeline steps.

# In[ ]:


get_ipython().system('echo $TFX_IMAGE_URI')


# In[ ]:


get_ipython().system('gcloud builds submit --tag $TFX_IMAGE_URI . --timeout=15m --machine-type=e2-highcpu-8')


# ### Compile pipeline

# In[ ]:


from src.tfx_pipelines import runner

pipeline_definition_file = f'{config.PIPELINE_NAME}.json'
pipeline_definition = runner.compile_prediction_pipeline(pipeline_definition_file)


# ### Submit run to Vertex Pipelines

# In[ ]:


from kfp.v2.google.client import AIPlatformClient

pipeline_client = AIPlatformClient(
    project_id=PROJECT, region=REGION)
                 
pipeline_client.create_run_from_job_spec(
    job_spec_path=pipeline_definition_file
)


# In[ ]:




