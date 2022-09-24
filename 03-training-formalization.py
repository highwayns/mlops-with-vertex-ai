#!/usr/bin/env python
# coding: utf-8

# # 03 - TFX Interactive Training Pipeline Execution
# 
# The purpose of this notebook is to interactively run the following `TFX` pipeline steps:
# 1. Receive hyperparameters using `hyperparam_gen` custom Python component.
# 2. Extract data from BigQuery using `BigQueryExampleGen` component.
# 3. Validate the raw data using `StatisticsGen` and `ExampleValidator` components.
# 4. Process the data using `Transform` component.
# 5. Train a custom model using `Trainer` component.
# 7. Evaluate and Validate the custom model using `ModelEvaluator` component.
# 7. Save the blessed to model registry location using `Pusher` component.
# 8. Upload the model to Vertex AI using `vertex_model_pusher` custom Python component
# 
# The custom components are implemented in the [tfx_pipeline/components.py](tfx_pipeline/components) module.

# ## Setup

# ### Import libraries

# In[ ]:


import os
import json
import numpy as np
import tfx
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_data_validation as tfdv
import tensorflow_model_analysis as tfma
from tensorflow_transform.tf_metadata import schema_utils
import logging

from src.common import features
from src.model_training import data
from src.tfx_pipelines import components

logging.getLogger().setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')

print("TFX Version:", tfx.__version__)
print("Tensorflow Version:", tf.__version__)


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
    
PARENT = f"projects/{PROJECT}/locations/{REGION}"
    
print("Project ID:", PROJECT)
print("Region:", REGION)
print("Bucket name:", BUCKET)
print("Service Account:", SERVICE_ACCOUNT)
print("Vertex API Parent URI:", PARENT)


# ### Set configurations

# In[ ]:


VERSION = 'v01'
DATASET_DISPLAY_NAME = 'chicago-taxi-tips'
MODEL_DISPLAY_NAME = f'{DATASET_DISPLAY_NAME}-classifier-{VERSION}'

WORKSPACE = f'gs://{BUCKET}/{DATASET_DISPLAY_NAME}'
RAW_SCHEMA_DIR = 'src/raw_schema'

MLMD_SQLLITE = 'mlmd.sqllite'
ARTIFACT_STORE = os.path.join(WORKSPACE, 'tfx_artifacts_interactive')
MODEL_REGISTRY = os.path.join(WORKSPACE, 'model_registry')
PIPELINE_NAME = f'{MODEL_DISPLAY_NAME}-train-pipeline'
PIPELINE_ROOT = os.path.join(ARTIFACT_STORE, PIPELINE_NAME)


# ## Create Interactive Context

# In[ ]:


REMOVE_ARTIFACTS = True

if tf.io.gfile.exists(ARTIFACT_STORE) and REMOVE_ARTIFACTS:
    print("Removing previous artifacts...")
    tf.io.gfile.rmtree(ARTIFACT_STORE)
    
if tf.io.gfile.exists(MLMD_SQLLITE) and REMOVE_ARTIFACTS:
    print("Deleting previous mlmd.sqllite...")
    tf.io.gfile.rmtree(MLMD_SQLLITE)
    
print(f'Pipeline artifacts directory: {PIPELINE_ROOT}')
print(f'Local metadata SQLlit path: {MLMD_SQLLITE}')


# In[ ]:


import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.sqlite.filename_uri = MLMD_SQLLITE
connection_config.sqlite.connection_mode = 3 # READWRITE_OPENCREATE
mlmd_store = mlmd.metadata_store.MetadataStore(connection_config)

context = InteractiveContext(
  pipeline_name=PIPELINE_NAME,
  pipeline_root=PIPELINE_ROOT,
  metadata_connection_config=connection_config
)


# ## 1. Hyperparameter generation

# In[ ]:


hyperparams_gen = components.hyperparameters_gen(
    num_epochs=5,
    learning_rate=0.001,
    batch_size=512,
    hidden_units='64,64',
)

context.run(hyperparams_gen, enable_cache=False)


# In[ ]:


json.load(
    tf.io.gfile.GFile(
        os.path.join(
            hyperparams_gen.outputs['hyperparameters'].get()[0].uri, 'hyperparameters.json')
    )
)


# ## 2. Data extraction

# In[ ]:


from src.common import datasource_utils
from tfx.extensions.google_cloud_big_query.example_gen.component import BigQueryExampleGen
from tfx.proto import example_gen_pb2, transform_pb2


# ### Extract train and eval splits

# In[ ]:


sql_query = datasource_utils.get_training_source_query(
    PROJECT, REGION, DATASET_DISPLAY_NAME, ml_use='UNASSIGNED', limit=5000)

output_config = example_gen_pb2.Output(
    split_config=example_gen_pb2.SplitConfig(
        splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=4),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=1),
        ]
    )
)

train_example_gen = BigQueryExampleGen(query=sql_query, output_config=output_config)

beam_pipeline_args=[
    f"--project={PROJECT}",
    f"--temp_location={os.path.join(WORKSPACE, 'tmp')}"
]

context.run(
    train_example_gen,
    beam_pipeline_args=beam_pipeline_args,
    enable_cache=False
)


# ### Extract test split

# In[ ]:


sql_query = datasource_utils.get_training_source_query(
    PROJECT, REGION, DATASET_DISPLAY_NAME, ml_use='TEST', limit=1000)

output_config = example_gen_pb2.Output(
    split_config=example_gen_pb2.SplitConfig(
        splits=[
            example_gen_pb2.SplitConfig.Split(name="test", hash_buckets=1),
        ]
    )
)

test_example_gen = BigQueryExampleGen(query=sql_query, output_config=output_config)

beam_pipeline_args=[
    f"--project={PROJECT}",
    f"--temp_location={os.path.join(WORKSPACE, 'tmp')}"
]

context.run(
    test_example_gen,
    beam_pipeline_args=beam_pipeline_args,
    enable_cache=False
)


# ### Read sample extract tfrecords

# In[ ]:


train_uri = os.path.join(train_example_gen.outputs['examples'].get()[0].uri, "Split-train/*")
source_raw_schema = tfdv.load_schema_text(os.path.join(RAW_SCHEMA_DIR, 'schema.pbtxt'))
raw_feature_spec = schema_utils.schema_as_feature_spec(source_raw_schema).feature_spec

def _parse_tf_example(tfrecord):
    return tf.io.parse_single_example(tfrecord, raw_feature_spec)

tfrecord_filenames = tf.data.Dataset.list_files(train_uri)
dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")
dataset = dataset.map(_parse_tf_example)

for raw_features in dataset.shuffle(1000).batch(3).take(1):
    for key in raw_features:
        print(f"{key}: {np.squeeze(raw_features[key], -1)}")
    print("")


# ## 3. Data validation

# ### Import raw schema

# In[ ]:


schema_importer = tfx.dsl.components.common.importer.Importer(
    source_uri=RAW_SCHEMA_DIR,
    artifact_type=tfx.types.standard_artifacts.Schema,
    reimport=False
)

context.run(schema_importer)


# ### Generate statistics

# In[ ]:


statistics_gen = tfx.components.StatisticsGen(
    examples=train_example_gen.outputs['examples'])
context.run(statistics_gen)


# In[ ]:


get_ipython().system('rm -r {RAW_SCHEMA_DIR}/.ipynb_checkpoints/')


# ### Validate statistics against schema

# In[ ]:


example_validator = tfx.components.ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_importer.outputs['result'],
)

context.run(example_validator)


# In[ ]:


context.show(example_validator.outputs['anomalies'])


# ## 4. Data transformation

# In[ ]:


_transform_module_file = 'src/preprocessing/transformations.py'

transform = tfx.components.Transform(
    examples=train_example_gen.outputs['examples'],
    schema=schema_importer.outputs['result'],
    module_file=_transform_module_file,
    splits_config=transform_pb2.SplitsConfig(
        analyze=['train'], transform=['train', 'eval']),
)

context.run(transform, enable_cache=False)


# ### Read sample transformed tfrecords

# In[ ]:


transformed_train_uri = os.path.join(transform.outputs['transformed_examples'].get()[0].uri, "Split-train/*")
transform_graph_uri = transform.outputs['transform_graph'].get()[0].uri

tft_output = tft.TFTransformOutput(transform_graph_uri)
transform_feature_spec = tft_output.transformed_feature_spec()

for input_features, target in data.get_dataset(
    transformed_train_uri, transform_feature_spec, batch_size=3).take(1):
    for key in input_features:
        print(f"{key} ({input_features[key].dtype}): {input_features[key].numpy().tolist()}")
    print(f"target: {target.numpy().tolist()}")


# ## 5. Model training

# In[ ]:


from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.experimental import latest_artifacts_resolver
from tfx.dsl.experimental import latest_blessed_model_resolver


# ### Get the latest model to warm start

# In[ ]:


latest_model_resolver = Resolver(
    strategy_class=latest_artifacts_resolver.LatestArtifactsResolver,
    latest_model=tfx.types.Channel(type=tfx.types.standard_artifacts.Model)
)

context.run(latest_model_resolver, enable_cache=False)


# ### Train the model

# In[ ]:


_train_module_file = 'src/model_training/runner.py'

trainer = tfx.components.Trainer(
    module_file=_train_module_file,
    examples=transform.outputs['transformed_examples'],
    schema=schema_importer.outputs['result'],
    base_model=latest_model_resolver.outputs['latest_model'],
    transform_graph=transform.outputs['transform_graph'],
    hyperparameters=hyperparams_gen.outputs['hyperparameters'],
)

context.run(trainer, enable_cache=False)


# ## 6. Model evaluation

# ### Get the latest blessed model for model validation.

# In[ ]:


blessed_model_resolver = Resolver(
    strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
    model=tfx.types.Channel(type=tfx.types.standard_artifacts.Model),
    model_blessing=tfx.types.Channel(type=tfx.types.standard_artifacts.ModelBlessing)
)

context.run(blessed_model_resolver, enable_cache=False)


# ### Evaluate and validate the model against the baseline model.

# In[ ]:


from tfx.components import Evaluator


# In[ ]:


eval_config = tfma.EvalConfig(
    model_specs=[
        tfma.ModelSpec(
            signature_name='serving_tf_example',
            label_key=features.TARGET_FEATURE_NAME,
            prediction_key='probabilities')
    ],
    slicing_specs=[
        tfma.SlicingSpec(),
    ],
    metrics_specs=[
        tfma.MetricsSpec(
            metrics=[   
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(
                    class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.8}),
                        # Change threshold will be ignored if there is no
                        # baseline model resolved from MLMD (first run).
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -1e-10}))),
        ])
    ])


evaluator = Evaluator(
    examples=test_example_gen.outputs['examples'],
    example_splits=['test'],
    model=trainer.outputs['model'],
    baseline_model=blessed_model_resolver.outputs['model'],
    eval_config=eval_config,
    schema=schema_importer.outputs['result']
)

context.run(evaluator, enable_cache=False)


# In[ ]:


evaluation_results = evaluator.outputs['evaluation'].get()[0].uri
print("validation_ok:", tfma.load_validation_result(evaluation_results).validation_ok, '\n')

for entry in list(tfma.load_metrics(evaluation_results))[0].metric_keys_and_values:
    value = entry.value.double_value.value
    if value:
        print(entry.key.name, ":", round(entry.value.double_value.value, 3))


# ## 7. Model pushing

# In[ ]:


exported_model_location = os.path.join(MODEL_REGISTRY, MODEL_DISPLAY_NAME)

push_destination=tfx.proto.pusher_pb2.PushDestination(
    filesystem=tfx.proto.pusher_pb2.PushDestination.Filesystem(
        base_directory=exported_model_location,
    )
)

pusher = tfx.components.Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=push_destination
)

context.run(pusher, enable_cache=False)


# ## 8. Model Upload to Vertex AI

# In[ ]:


serving_runtime = 'tf2-cpu.2-5'
serving_image_uri = f"us-docker.pkg.dev/vertex-ai/prediction/{serving_runtime}:latest"

labels = {
    'dataset_name': DATASET_DISPLAY_NAME,
    'pipeline_name': PIPELINE_NAME
}
labels = json.dumps(labels)

vertex_model_uploader = components.vertex_model_uploader(
    project=PROJECT,
    region=REGION,
    model_display_name=MODEL_DISPLAY_NAME,
    pushed_model_location=exported_model_location,
    serving_image_uri=serving_image_uri,
    model_blessing=evaluator.outputs['blessing'],
    explanation_config='',
    labels=labels
)

context.run(vertex_model_uploader, enable_cache=False)


# In[ ]:


vertex_model_uploader.outputs['uploaded_model'].get()[0].get_string_custom_property('model_uri')


# In[ ]:




