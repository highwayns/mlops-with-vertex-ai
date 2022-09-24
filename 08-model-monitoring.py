#!/usr/bin/env python
# coding: utf-8

# # 08 - Model Monitoring
# 
# This notebook covers configuring model monitoring jobs for skew and drift detection:
# 1. Set skew and drift threshold.
# 2. Create a monitoring job for all the models under and endpoint.
# 3. List the monitoring jobs.
# 4. Simulate skewed prediction requests.
# 5. Pause and delete the monitoring job.

# ## Setup

# ### Import libraries

# In[ ]:


import copy
from datetime import datetime
import time

from google.protobuf.duration_pb2 import Duration
from google.cloud import aiplatform as vertex_ai
from google.cloud import aiplatform_v1beta1 as vertex_ai_beta


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

PARENT = f"projects/{PROJECT}/locations/{REGION}"

print("Project ID:", PROJECT)
print("Region:", REGION)
print("Bucket name:", BUCKET)
print("Vertex API Parent URI:", PARENT)


# ### Set configurations

# In[ ]:


DATASET_DISPLAY_NAME = 'chicago-taxi-tips'
ENDPOINT_DISPLAY_NAME = 'chicago-taxi-tips-classifier'
MONITORING_JOB_NAME = f"monitor-{ENDPOINT_DISPLAY_NAME}"
NOTIFY_EMAILS = ["<your-email-address>"] # Change to your email address.

LOG_SAMPLE_RATE = 0.8
MONITOR_INTERVAL = 3600
TARGET_FEATURE_NAME = 'tip_bin'


# ## Create Job Service Client

# In[ ]:


job_client_beta = vertex_ai_beta.JobServiceClient(
    client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}
)


# ## 1. Set Skew and Drift Thresholds

# In[ ]:


SKEW_THRESHOLDS = {
    'trip_month': 0.3,
    'trip_day': 0.3,
    'trip_day_of_week': 0.3,
    'trip_hour': 0.3,
    'trip_seconds': 0.3,
    'trip_miles': 0.3,
    'payment_type': 0.3,
    'pickup_grid': 0.3,
    'dropoff_grid': 0.3,
    'euclidean': 0.3,
    'loc_cross': 0.3,  
}

DRIFT_THRESHOLDS = {
    'trip_month': 0.3,
    'trip_day': 0.3,
    'trip_day_of_week': 0.3,
    'trip_hour': 0.3,
    'trip_seconds': 0.3,
    'trip_miles': 0.3,
    'payment_type': 0.3,
    'pickup_grid': 0.3,
    'dropoff_grid': 0.3,
    'euclidean': 0.3,
    'loc_cross': 0.3,  
}


# ## 2. Create Monitoring Job

# ### Retrieve the Vertex dataset and endpoint models to monitor

# In[ ]:


dataset = vertex_ai.TabularDataset.list(
    filter=f"display_name={DATASET_DISPLAY_NAME}", 
    order_by="update_time")[-1]

bq_source_uri = dataset.gca_resource.metadata["inputConfig"]["bigquerySource"]["uri"]
    
endpoint = vertex_ai.Endpoint.list(
    filter=f'display_name={ENDPOINT_DISPLAY_NAME}', 
    order_by="update_time")[-1]

endpoint_uri = endpoint.gca_resource.name

model_ids = [model.id for model in endpoint.list_models()]


# ### Configure the monitoring job

# In[ ]:


skew_thresholds = {
    feature: vertex_ai_beta.ThresholdConfig(value=float(value))
    for feature, value in SKEW_THRESHOLDS.items()
}

drift_thresholds = {
    feature: vertex_ai_beta.ThresholdConfig(value=float(value))
    for feature, value in DRIFT_THRESHOLDS.items()
}

skew_config = vertex_ai_beta.ModelMonitoringObjectiveConfig.TrainingPredictionSkewDetectionConfig(
    skew_thresholds=skew_thresholds
)

drift_config = vertex_ai_beta.ModelMonitoringObjectiveConfig.PredictionDriftDetectionConfig(
    drift_thresholds=drift_thresholds
)

sampling_config = vertex_ai_beta.SamplingStrategy(
    random_sample_config=vertex_ai_beta.SamplingStrategy.RandomSampleConfig(
        sample_rate=LOG_SAMPLE_RATE
    )
)

schedule_config = vertex_ai_beta.ModelDeploymentMonitoringScheduleConfig(
    monitor_interval=Duration(seconds=MONITOR_INTERVAL)
)

training_dataset = vertex_ai_beta.ModelMonitoringObjectiveConfig.TrainingDataset(
    target_field=TARGET_FEATURE_NAME,
    bigquery_source = vertex_ai_beta.types.io.BigQuerySource(
        input_uri=bq_source_uri
    )
)


objective_template = vertex_ai_beta.ModelDeploymentMonitoringObjectiveConfig(
    objective_config=vertex_ai_beta.ModelMonitoringObjectiveConfig(
        training_dataset=training_dataset,
        training_prediction_skew_detection_config=skew_config,
        prediction_drift_detection_config=drift_config,
    )
)

deployment_objective_configs = []
for model_id in model_ids:
    objective_config = copy.deepcopy(objective_template)
    objective_config.deployed_model_id = model_id
    deployment_objective_configs.append(objective_config)

alerting_config = vertex_ai_beta.ModelMonitoringAlertConfig(
    email_alert_config=vertex_ai_beta.ModelMonitoringAlertConfig.EmailAlertConfig(
        user_emails=NOTIFY_EMAILS
    )
)


# ### Instantiate a monitoring job

# In[ ]:


job = vertex_ai_beta.ModelDeploymentMonitoringJob(
    display_name=MONITORING_JOB_NAME,
    endpoint=endpoint_uri,
    model_deployment_monitoring_objective_configs=deployment_objective_configs,
    logging_sampling_strategy=sampling_config,
    model_deployment_monitoring_schedule_config=schedule_config,
    model_monitoring_alert_config=alerting_config,
)


# ### Submit the job for creation

# In[ ]:


response = job_client_beta.create_model_deployment_monitoring_job(
    parent=PARENT, model_deployment_monitoring_job=job
)
response


# ## 3. List Monitoring Jobs

# In[ ]:


monitoring_jobs = job_client_beta.list_model_deployment_monitoring_jobs(parent=PARENT)
monitoring_job = [entry for entry in monitoring_jobs if entry.display_name == MONITORING_JOB_NAME][0]
monitoring_job


# ## 4. Simulate skewed prediction requests

# In[ ]:


num_requests = 100

print("Simulation started...")
for idx in range(num_requests):
    request = [{
        "dropoff_grid": ["POINT(-87.6 41.9)"],
        "euclidean": [2064.2696],
        "loc_cross": [""],
        "payment_type": ["Credit Card"],
        "pickup_grid": ["POINT(-87.6 41.9)"],
        "trip_miles": [1.37],
        "trip_day": [int(random.uniform(10, 50))],
        "trip_hour": [int(random.uniform(10, 50))],
        "trip_month": [int(random.uniform(1, 10))],
        "trip_day_of_week": [int(random.uniform(1, 7))],
        "trip_seconds": [int(random.uniform(60, 600))]
    }]
    
    endpoint.predict(request)
    time.sleep(0.5)
    
    if idx % 10 == 0:
        print(f'{idx + 1} of {num_requests} prediction requests were invoked.')
print("Simulation finished.")


# ## 5. Pause Monitoring Job

# In[ ]:


job_client_beta.pause_model_deployment_monitoring_job(name=monitoring_job.name)


# ## Delete Monitoring Job

# In[ ]:


job_client_beta.delete_model_deployment_monitoring_job(name=monitoring_job.name)


# In[ ]:




