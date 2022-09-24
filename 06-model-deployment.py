#!/usr/bin/env python
# coding: utf-8

# # 06 - Model Deployment
# 
# The purpose of this notebook is to execute a CI/CD routine to test and deploy the trained model to `Vertex AI` as an `Endpoint` for online prediction serving. The notebook covers the following steps:
# 1. Run the test steps locally.
# 2. Execute the model deployment `CI/CD` steps using `Cloud Build`.
# 
# 

# ## Setup

# ### Import libraries

# In[ ]:


import os
import logging

logging.getLogger().setLevel(logging.INFO)


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


VERSION = 'v01'
DATASET_DISPLAY_NAME = 'chicago-taxi-tips'
MODEL_DISPLAY_NAME = f'{DATASET_DISPLAY_NAME}-classifier-{VERSION}'
ENDPOINT_DISPLAY_NAME = f'{DATASET_DISPLAY_NAME}-classifier'

CICD_IMAGE_NAME = 'cicd:latest'
CICD_IMAGE_URI = f"gcr.io/{PROJECT}/{CICD_IMAGE_NAME}"


# ## 1. Run CI/CD steps locally

# In[ ]:


os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION
os.environ['MODEL_DISPLAY_NAME'] = MODEL_DISPLAY_NAME
os.environ['ENDPOINT_DISPLAY_NAME'] = ENDPOINT_DISPLAY_NAME


# ### Run the model artifact testing

# In[ ]:


get_ipython().system('py.test src/tests/model_deployment_tests.py::test_model_artifact -s')


# ### Run create endpoint

# In[ ]:


get_ipython().system('python build/utils.py     --mode=create-endpoint    --project={PROJECT}    --region={REGION}    --endpoint-display-name={ENDPOINT_DISPLAY_NAME}')


# ### Run deploy model

# In[ ]:


get_ipython().system('python build/utils.py     --mode=deploy-model    --project={PROJECT}    --region={REGION}    --endpoint-display-name={ENDPOINT_DISPLAY_NAME}    --model-display-name={MODEL_DISPLAY_NAME}')


# ### Test deployed model endpoint

# In[ ]:


get_ipython().system('py.test src/tests/model_deployment_tests.py::test_model_endpoint')


# ## 2. Execute the Model Deployment CI/CD routine in Cloud Build
# 
# The CI/CD routine is defined in the [model-deployment.yaml](model-deployment.yaml) file, and consists of the following steps:
# 1. Load and test the the trained model interface.
# 2. Create and endpoint in Vertex AI if it doesn't exists.
# 3. Deploy the model to the endpoint.
# 4. Test the endpoint.

# ### Build CI/CD container Image for Cloud Build
# 
# This is the runtime environment where the steps of testing and deploying model will be executed.

# In[ ]:


get_ipython().system('echo $CICD_IMAGE_URI')


# In[ ]:


get_ipython().system('gcloud builds submit --tag $CICD_IMAGE_URI build/. --timeout=15m')


# ### Run CI/CD from model deployment using Cloud Build

# In[ ]:


REPO_URL = "https://github.com/GoogleCloudPlatform/mlops-with-vertex-ai.git" # Change to your github repo.
BRANCH = "main" 


# In[ ]:


SUBSTITUTIONS=f"""_REPO_URL='{REPO_URL}',_BRANCH={BRANCH},_CICD_IMAGE_URI={CICD_IMAGE_URI},_PROJECT={PROJECT},_REGION={REGION},_MODEL_DISPLAY_NAME={MODEL_DISPLAY_NAME},_ENDPOINT_DISPLAY_NAME={ENDPOINT_DISPLAY_NAME},"""

get_ipython().system('echo $SUBSTITUTIONS')


# In[ ]:


get_ipython().system('gcloud builds submit --no-source --config build/model-deployment.yaml --substitutions {SUBSTITUTIONS} --timeout=30m')


# In[ ]:





# In[ ]:




