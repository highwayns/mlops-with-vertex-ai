#!/usr/bin/env python
# coding: utf-8

# # 04 - Test and Deploy Training Pipeline to Vertex Pipelines
# 
# The purpose of this notebook is to test, deploy, and run the `TFX` pipeline on `Vertex Pipelines`. The notebook covers the following tasks:
# 1. Run the tests locally.
# 2. Run the pipeline using `Vertex Pipelines`
# 3. Execute the pipeline deployment `CI/CD` steps using `Cloud Build`.

# ## Setup

# ### Import libraries

# In[ ]:


import os
import kfp
import tfx

print("Tensorflow Version:", tfx.__version__)
print("KFP Version:", kfp.__version__)


# ### Setup Google Cloud project

# In[ ]:


PROJECT = '[your-project-id]' # Change to your project id.
REGION = 'us-central1' # Change to your region.
BUCKET =  '[your-bucket-name]' # Change to your bucket name.
SERVICE_ACCOUNT = "[your-service-account]"

if PROJECT == "" or PROJECT is None or PROJECT == "[your-project-id]":
    # Get your GCP project id from gcloud
    shell_output = get_ipython().getoutput("gcloud config list --format 'value(core.project)' 2>/dev/null")
    PROJECT = shell_output[0]
    
if SERVICE_ACCOUNT == "" or SERVICE_ACCOUNT is None or SERVICE_ACCOUNT == "[your-service-account]":
    # Get your GCP project id from gcloud
    shell_output = get_ipython().getoutput("gcloud config list --format 'value(core.account)' 2>/dev/null")
    SERVICE_ACCOUNT = shell_output[0]
    
if BUCKET == "" or BUCKET is None or BUCKET == "[your-bucket-name]":
    # Get your bucket name to GCP project id
    BUCKET = PROJECT
    # Try to create the bucket if it doesn't exists
    get_ipython().system(' gsutil mb -l $REGION gs://$BUCKET')
    print("")
    
print("Project ID:", PROJECT)
print("Region:", REGION)
print("Bucket name:", BUCKET)
print("Service Account:", SERVICE_ACCOUNT)


# ### Set configurations

# In[ ]:


BQ_LOCATION = 'US'
BQ_DATASET_NAME = 'playground_us' # Change to your BQ dataset name.
BQ_TABLE_NAME = 'chicago_taxitrips_prep'

VERSION = 'v01'
DATASET_DISPLAY_NAME = 'chicago-taxi-tips'
MODEL_DISPLAY_NAME = f'{DATASET_DISPLAY_NAME}-classifier-{VERSION}'
PIPELINE_NAME = f'{MODEL_DISPLAY_NAME}-train-pipeline'

CICD_IMAGE_NAME = 'cicd:latest'
CICD_IMAGE_URI = f"gcr.io/{PROJECT}/{CICD_IMAGE_NAME}"


# In[ ]:


get_ipython().system('rm -r src/raw_schema/.ipynb_checkpoints/')


# ## 1. Run the CICD steps locally

# ### Set pipeline configurations for the local run

# In[ ]:


os.environ["DATASET_DISPLAY_NAME"] = DATASET_DISPLAY_NAME
os.environ["MODEL_DISPLAY_NAME"] =  MODEL_DISPLAY_NAME
os.environ["PIPELINE_NAME"] = PIPELINE_NAME
os.environ["PROJECT"] = PROJECT
os.environ["REGION"] = REGION
os.environ["BQ_LOCATION"] = BQ_LOCATION
os.environ["BQ_DATASET_NAME"] = BQ_DATASET_NAME
os.environ["BQ_TABLE_NAME"] = BQ_TABLE_NAME
os.environ["GCS_LOCATION"] = f"gs://{BUCKET}/{DATASET_DISPLAY_NAME}/e2e_tests"
os.environ["TRAIN_LIMIT"] = "1000"
os.environ["TEST_LIMIT"] = "100"
os.environ["UPLOAD_MODEL"] = "0"
os.environ["ACCURACY_THRESHOLD"] = "0.1"
os.environ["BEAM_RUNNER"] = "DirectRunner"
os.environ["TRAINING_RUNNER"] = "local"


# In[ ]:


from src.tfx_pipelines import config
import importlib
importlib.reload(config)

for key, value in config.__dict__.items():
    if key.isupper(): print(f'{key}: {value}')


# ### Run unit tests

# In[ ]:


get_ipython().system('py.test src/tests/datasource_utils_tests.py -s')


# In[ ]:


get_ipython().system('py.test src/tests/model_tests.py -s')


# ### Run e2e pipeline test

# In[ ]:


get_ipython().system('py.test src/tests/pipeline_deployment_tests.py::test_e2e_pipeline -s')


# ## 2. Run the training pipeline using Vertex Pipelines
# 
# 

# ### Set the pipeline configurations for the Vertex AI run

# In[ ]:


os.environ["DATASET_DISPLAY_NAME"] = DATASET_DISPLAY_NAME
os.environ["MODEL_DISPLAY_NAME"] = MODEL_DISPLAY_NAME
os.environ["PIPELINE_NAME"] = PIPELINE_NAME
os.environ["PROJECT"] = PROJECT
os.environ["REGION"] = REGION
os.environ["GCS_LOCATION"] = f"gs://{BUCKET}/{DATASET_DISPLAY_NAME}"
os.environ["TRAIN_LIMIT"] = "85000"
os.environ["TEST_LIMIT"] = "15000"
os.environ["BEAM_RUNNER"] = "DataflowRunner"
os.environ["TRAINING_RUNNER"] = "vertex"
os.environ["TFX_IMAGE_URI"] = f"gcr.io/{PROJECT}/{DATASET_DISPLAY_NAME}:{VERSION}"
os.environ["ENABLE_CACHE"] = "1"


# In[ ]:


from src.tfx_pipelines import config
import importlib
importlib.reload(config)

for key, value in config.__dict__.items():
    if key.isupper(): print(f'{key}: {value}')


# ### Build the ML container image
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
pipeline_definition = runner.compile_training_pipeline(pipeline_definition_file)


# In[ ]:


PIPELINES_STORE = f"gs://{BUCKET}/{DATASET_DISPLAY_NAME}/compiled_pipelines/"
get_ipython().system('gsutil cp {pipeline_definition_file} {PIPELINES_STORE}')


# ### Submit run to Vertex Pipelines

# In[ ]:


from kfp.v2.google.client import AIPlatformClient

pipeline_client = AIPlatformClient(
    project_id=PROJECT, region=REGION)
                 
job = pipeline_client.create_run_from_job_spec(
    job_spec_path=pipeline_definition_file,
    parameter_values={
        'learning_rate': 0.003,
        'batch_size': 512,
        'hidden_units': '128,128',
        'num_epochs': 30,
    }
)


# ### Extracting pipeline runs metadata

# In[ ]:


from google.cloud import aiplatform as vertex_ai

pipeline_df = vertex_ai.get_pipeline_df(PIPELINE_NAME)
pipeline_df = pipeline_df[pipeline_df.pipeline_name == PIPELINE_NAME]
pipeline_df.T


# ## 3. Execute the pipeline deployment CI/CD steps in Cloud Build
# 
# The CI/CD routine is defined in the [pipeline-deployment.yaml](build/pipeline-deployment.yaml) file, and consists of the following steps:
# 1. Clone the repository to the build environment.
# 2. Run unit tests.
# 3. Run a local e2e test of the pipeline.
# 4. Build the ML container image for pipeline steps.
# 5. Compile the pipeline.
# 6. Upload the pipeline to Cloud Storage.

# ### Build CI/CD container Image for Cloud Build
# 
# This is the runtime environment where the steps of testing and deploying the pipeline will be executed.

# In[ ]:


get_ipython().system('echo $CICD_IMAGE_URI')


# In[ ]:


get_ipython().system('gcloud builds submit --tag $CICD_IMAGE_URI build/. --timeout=15m --machine-type=e2-highcpu-8')


# ### Run CI/CD from pipeline deployment using Cloud Build

# In[ ]:


REPO_URL = "https://github.com/GoogleCloudPlatform/mlops-with-vertex-ai.git" # Change to your github repo.
BRANCH = "main"

GCS_LOCATION = f"gs://{BUCKET}/{DATASET_DISPLAY_NAME}/"
TEST_GCS_LOCATION = f"gs://{BUCKET}/{DATASET_DISPLAY_NAME}/e2e_tests"
CI_TRAIN_LIMIT = 1000
CI_TEST_LIMIT = 100
CI_UPLOAD_MODEL = 0
CI_ACCURACY_THRESHOLD = 0.1
BEAM_RUNNER = "DataflowRunner"
TRAINING_RUNNER = "vertex"
VERSION = 'tfx-1.2'
PIPELINE_NAME = f'{MODEL_DISPLAY_NAME}-train-pipeline'
PIPELINES_STORE = os.path.join(GCS_LOCATION, "compiled_pipelines")

TFX_IMAGE_URI = f"gcr.io/{PROJECT}/{DATASET_DISPLAY_NAME}:{VERSION}"

SUBSTITUTIONS=f"""_REPO_URL='{REPO_URL}',_BRANCH={BRANCH},_CICD_IMAGE_URI={CICD_IMAGE_URI},_PROJECT={PROJECT},_REGION={REGION},_GCS_LOCATION={GCS_LOCATION},_TEST_GCS_LOCATION={TEST_GCS_LOCATION},_BQ_LOCATION={BQ_LOCATION},_BQ_DATASET_NAME={BQ_DATASET_NAME},_BQ_TABLE_NAME={BQ_TABLE_NAME},_DATASET_DISPLAY_NAME={DATASET_DISPLAY_NAME},_MODEL_DISPLAY_NAME={MODEL_DISPLAY_NAME},_CI_TRAIN_LIMIT={CI_TRAIN_LIMIT},_CI_TEST_LIMIT={CI_TEST_LIMIT},_CI_UPLOAD_MODEL={CI_UPLOAD_MODEL},_CI_ACCURACY_THRESHOLD={CI_ACCURACY_THRESHOLD},_BEAM_RUNNER={BEAM_RUNNER},_TRAINING_RUNNER={TRAINING_RUNNER},_TFX_IMAGE_URI={TFX_IMAGE_URI},_PIPELINE_NAME={PIPELINE_NAME},_PIPELINES_STORE={PIPELINES_STORE}"""

get_ipython().system('echo $SUBSTITUTIONS')


# In[ ]:


get_ipython().system('gcloud builds submit --no-source --timeout=60m --config build/pipeline-deployment.yaml --substitutions {SUBSTITUTIONS} --machine-type=e2-highcpu-8')


# In[ ]:




