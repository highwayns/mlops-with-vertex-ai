#!/usr/bin/env python
# coding: utf-8

# # 05 - Continuous Training
# 
# After testing, compiling, and uploading the pipeline definition to Cloud Storage, the pipeline is executed with respect to a trigger. We use [Cloud Functions](https://cloud.google.com/functions) and [Cloud Pub/Sub](https://cloud.google.com/pubsub) as a triggering mechanism. The triggering can be scheduled using [Cloud Scheduler](https://cloud.google.com/scheduler). The trigger source sends a message to a Cloud Pub/Sub topic that the Cloud Function listens to, and then it submits the pipeline to AI Platform Managed Pipelines to be executed.
# 
# This notebook covers the following steps:
# 1. Create the Cloud Pub/Sub topic.
# 2. Deploy the Cloud Function 
# 3. Test triggering a pipeline.
# 4. Extracting pipeline run metadata.

# ## Setup

# ### Import libraries

# In[ ]:


import json
import os
import logging
import tensorflow as tf
import tfx
import IPython 

logging.getLogger().setLevel(logging.INFO)

print("Tensorflow Version:", tfx.__version__)


# ### Setup Google Cloud project

# In[ ]:


PROJECT = '[your-project-id]' # Change to your project id.
REGION = 'us-central1' # Change to your region.
BUCKET =  '[your-bucket-name]' # Change to your bucket name.

if PROJECT == "" or PROJECT is None or PROJECT == "[your-project-id]":
    # Get your GCP project id from gcloud
    shell_output = get_ipython().getoutput("gcloud config list --format 'value(core.project)' 2>/dev/null")
    PROJECT = shell_output[0]
    
if BUCKET == "" or BUCKET is None or BUCKET == "[your-bucket-name]":
    # Get your bucket name to GCP projet id
    BUCKET = PROJECT

print("Project ID:", PROJECT)
print("Region:", REGION)
print("Bucket name:", BUCKET)


# ### Set configurations

# In[ ]:


VERSION = 'v01'
DATASET_DISPLAY_NAME = 'chicago-taxi-tips'
MODEL_DISPLAY_NAME = f'{DATASET_DISPLAY_NAME}-classifier-{VERSION}'
PIPELINE_NAME = f'{MODEL_DISPLAY_NAME}-train-pipeline'

PIPELINES_STORE = f'gs://{BUCKET}/{DATASET_DISPLAY_NAME}/compiled_pipelines/'
GCS_PIPELINE_FILE_LOCATION = os.path.join(PIPELINES_STORE, f'{PIPELINE_NAME}.json')
PUBSUB_TOPIC = f'trigger-{PIPELINE_NAME}'
CLOUD_FUNCTION_NAME = f'trigger-{PIPELINE_NAME}-fn'


# In[ ]:


get_ipython().system('gsutil ls {GCS_PIPELINE_FILE_LOCATION}')


# ## 1. Create a Pub/Sub topic

# In[ ]:


get_ipython().system('gcloud pubsub topics create {PUBSUB_TOPIC}')


# ## 2. Deploy the Cloud Function

# In[ ]:


ENV_VARS=f"""PROJECT={PROJECT},REGION={REGION},GCS_PIPELINE_FILE_LOCATION={GCS_PIPELINE_FILE_LOCATION}
"""

get_ipython().system('echo {ENV_VARS}')


# In[ ]:


get_ipython().system('rm -r src/pipeline_triggering/.ipynb_checkpoints')


# In[ ]:


get_ipython().system('gcloud functions deploy {CLOUD_FUNCTION_NAME}     --region={REGION}     --trigger-topic={PUBSUB_TOPIC}     --runtime=python37     --source=src/pipeline_triggering    --entry-point=trigger_pipeline    --stage-bucket={BUCKET}    --update-env-vars={ENV_VARS}')


# In[ ]:


cloud_fn_url = f"https://console.cloud.google.com/functions/details/{REGION}/{CLOUD_FUNCTION_NAME}"
html = f'See the Cloud Function details <a href="{cloud_fn_url}" target="_blank">here</a>.'
IPython.display.display(IPython.display.HTML(html))


# ## 3. Trigger the pipeline

# In[ ]:


from google.cloud import pubsub

publish_client = pubsub.PublisherClient()
topic = f'projects/{PROJECT}/topics/{PUBSUB_TOPIC}'
data = {
    'num_epochs': 7,
    'learning_rate': 0.0015,
    'batch_size': 512,
    'hidden_units': '256,126'
}
message = json.dumps(data)

_ = publish_client.publish(topic, message.encode())


# Wait for a few seconds for the pipeline run to be submitted, then you can see the run in the Cloud Console

# In[ ]:


from kfp.v2.google.client import AIPlatformClient

pipeline_client = AIPlatformClient(
    project_id=PROJECT, region=REGION)
 
job_display_name = pipeline_client.list_jobs()['pipelineJobs'][0]['displayName']
job_url = f"https://console.cloud.google.com/vertex-ai/locations/{REGION}/pipelines/runs/{job_display_name}"
html = f'See the Pipeline job <a href="{job_url}" target="_blank">here</a>.'
IPython.display.display(IPython.display.HTML(html))


# ## 4. Extracting pipeline runs metadata

# In[ ]:


from google.cloud import aiplatform as vertex_ai

pipeline_df = vertex_ai.get_pipeline_df(PIPELINE_NAME)
pipeline_df = pipeline_df[pipeline_df.pipeline_name == PIPELINE_NAME]
pipeline_df.T


# In[ ]:




