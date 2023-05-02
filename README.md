# Azure AutoML SDK Guide for Beginners
Welcome to this guide on how to run your first AutoML project on Azure using the Python SDK! In this tutorial, we’ll be focusing on multilabel classification with text data using AutoML NLP. AutoML, or automated machine learning, is a rapidly growing field that enables developers to create machine learning models with minimal effort and technical expertise. With the help of the Azure platform and Python SDK, we’ll be exploring how to build a multilabel classification model using AutoML NLP. We will be woring [arXiv Paper Abstracts](https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts) famous dataset. 

So, let’s get started and learn how to leverage the power of AutoML to build robust machine-learning models quickly and easily.

## Prerequisites
1. A resource group.
2. A workspace.
3. A subscription ID.
4. Installed and imported those libraries:
``` python
# pip install azure-identity
from azure.identity import DefaultAzureCredential 
# pip install azure-ai-ml
from azure.ai.ml.constants import AssetTypes 
from azure.ai.ml import automl, Input, MLClient
```

## 1. Connect to Azure Machine Learning Workspace
To connect to a workspace, we need identifier parameters: A subscription, resource group, and workspace name.

We will use these details in the MLClient from azure.ai.ml to get a handle on the required Azure Machine Learning workspace.

So, let's create our `ml_client` :
``` python
credentials = DefaultAzureCredential()
subscription_id = "<subscription_id>"
resource_group_name = "<resource_group_name>"
workspace = "<workspace>"
try:
    ml_client = MLClient(credentials, subscription_id, resource_group_name, workspace)
    print("MLClient created")
except Exception as ex:
    print(ex)
```
## 2. Data Preparation
You can get any dataset you want from the internet, in this tutorial, I will be using the [arxiv_data dataset](https://www.kaggle.com/spsayakpaul/arxiv-paper-abstracts).

So, our model will be trained on a dataset of paper titles and abstracts and their corresponding subject areas to classify/suggest what category corresponding papers could be best associated with.

Make sure to divide your dataset into 80% training, 10% validation, and 10% testing.

Put each partition into a separate directory and add the MLTable file within each directory like in this image:
![image](https://user-images.githubusercontent.com/56788883/235770822-69609f97-30fd-4ad4-8787-8dfed5545a59.png)


Where the MLTable file is:
``` yaml
paths:
  # change this for text.csv and train.csv
  - file: ./validation.csv 
transformations:
  - read_delimited:
      delimiter: ','
      encoding: 'utf8'
      empty_as_string: false
```
>> For documentation on creating your own MLTable assets for jobs beyond this toutrial:
>>1. [Details on how to write MLTable YAMLs](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-mltable) (required for each MLTable asset)
>>2. [How to work with them in the v2 CLI/SDK](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-data-assets?tabs=Python-SDK).

Finally, let’s read our data(you can read them in any way you like).
``` python
# MLTable folders
training_mltable_path = "./trainData/"
validation_mltable_path = "./validationData/"

# Training MLTable defined locally, with local data to be uploaded
my_training_data_input = Input(type=AssetTypes.MLTABLE, path=training_mltable_path)

# Validation MLTable defined locally, with local data to be uploaded
my_validation_data_input = Input(type=AssetTypes.MLTABLE, path=validation_mltable_path)
```

## 3. Configure the AutoML NLP Text Classification Multilabel training job
![image](https://user-images.githubusercontent.com/56788883/235770892-c8028031-12a0-4464-81a7-360c38ec8e29.png)

Now, we want to create or get an existing Azure Machine Learning compute target. The compute target is used for training machine learning models and can be thought of as a set of virtual machines that run in parallel to speed up the training process.
``` python
from azure.ai.ml.entities import AmlCompute
from azure.core.exceptions import ResourceNotFoundError

compute_name = "mwk"

try:
    _ = ml_client.compute.get(compute_name)
    print("Found existing compute target.")
except ResourceNotFoundError:
    print("Creating a new compute target...")
    compute_config = AmlCompute(
        name=compute_name,
        type="amlcompute",
        size="Standard_NC6", # a GPU compute target
        idle_time_before_scale_down=120,
        min_instances=0,
        max_instances=4,
    )
    # Finally, the new compute target is created using ml_client.begin_create_or_update(compute_config).result(). 
    # The .result() method ensures that the creation operation completes before moving on to the next step of the code.
    ml_client.begin_create_or_update(compute_config).result()
```
Now, we want to create a new text classification multilabel experiment using Azure Machine Learning’s automated machine learning (AutoML) functionality, with the specified configuration settings.
``` python
# Create the AutoML job with the related factory-function.
exp_name = "dpv2-nlp-multilabel"
exp_timeout = 120
text_classification_multilabel_job = automl.text_classification_multilabel(
    compute=compute_name,
    experiment_name=exp_name,
    training_data=trainData,
    validation_data=validationData,
    target_column_name="terms",
    primary_metric="accuracy", # specifies the evaluation metric to be used to compare the performance of different models during the AutoML experiment.
    tags={"Name": "BigData-Text-Classification-Multilabel"},
)
text_classification_multilabel_job.set_limits(timeout_minutes=exp_timeout)
```
After the AutoML experiment configuration is set up, text_classification_multilabel_job.set_limits(timeout_minutes=exp_timeout) is used to set the maximum amount of time that the experiment can run for.

Once the configuration is complete and the timeout is set, the AutoML experiment can be run using text_classification_multilabel_job.fit() to train multiple models and find the best-performing model based on the specified evaluation metric.

## 4. Run the AutoML NLP Text Classification Multilabel training job
Run:
``` python
returned_job = ml_client.jobs.create_or_update(
    text_classification_multilabel_job
)  # submit the job to the backend
print(f"Created job: {returned_job}")
```
The ml_client.jobs.create_or_update() method is called with the text_classification_multilabel_job object as an argument. This method creates a new job or updates an existing job with the specified experiment configuration. The create_or_update() method returns a job object that represents the job in the Azure Machine Learning service backend.

The returned_job variable is assigned to the job object returned by the create_or_update() method. The job object contains information about the job, such as its ID, status, and run history.

The job is then submitted to the backend for execution. The status of the job can be tracked and monitored using the Azure Machine Learning service backend.

To stream the logs and status updates for a specified job in real time:
``` python

ml_client.jobs.stream(returned_job.name) # The actual execution of the job is started using the ml_client.jobs.stream() method.
# It also prints a link to Microsoft Azure Machine Learning Studio to track the job
```
If you opened the output link, you can see your job running like this:
<!--  -->
## 5. Retrieve Model Information from the Best Trial of the Model
Once all the trials complete training, we can retrieve the best model.
``` python
# Obtain best child run id
returned_nlp_job = ml_client.jobs.get(name=returned_job.name)
best_child_run_id = returned_nlp_job.tags["automl_best_child_run_id"]

# Obtain the tracking URI for MLFlow
# pip install azureml-mlflow
import mlflow

# Obtain the tracking URL from MLClient
MLFLOW_TRACKING_URI = ml_client.workspaces.get(
    name=ml_client.workspace_name
).mlflow_tracking_uri
# Set the MLFLOW TRACKING URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print("\nCurrent tracking uri: {}".format(mlflow.get_tracking_uri()))
```
Get the AutoML parent Job
```python
# pip install azureml-mlflow
from mlflow.tracking.client import MlflowClient

# Get the AutoML parent Job
job_name = returned_job.name

# Get the parent run
mlflow_parent_run = mlflow_client.get_run(job_name)

print("Parent Run: ")
print(mlflow_parent_run)
```
Get the AutoML best child run:
```python
best_run = mlflow_client.get_run(best_child_run_id)
# OR
# best_child_run_id = mlflow_parent_run.data.tags["automl_best_child_run_id"]
print("Best child run: ")
print(best_run)

print("Best child run metrics: ")
print(best_run.data.metrics)
```
6. Download the best model locally
Access the results (such as Models, Artifacts, and Metrics) of a previously completed AutoML Run.
```python
import os
from mlflow.artifacts import download_artifacts

# Create local folder
local_dir = "./artifact_downloads"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)
# Download run's artifacts/outputs
local_path = download_artifacts(
    run_id=best_run.info.run_id, artifact_path="outputs", dst_path=local_dir
)
print("Artifacts downloaded in: {}".format(local_path))
print("Artifacts: {}".format(os.listdir(local_path)))
```
You can Show the contents of the MLFlow model folder using: `os.listdir(“./artifact_downloads/outputs/mlflow-model”)`

## Conclusion
In conclusion, I hope this guide has been helpful in providing an introduction to building a multilabel classification model with text data using AutoML NLP on Azure. With the power of automated machine learning and the convenience of the Python SDK, developers can create accurate and reliable models with minimal effort and technical expertise. By following the steps outlined in this tutorial, you can quickly build and deploy your first AutoML project on Azure, opening up a world of possibilities for your machine-learning applications. I encourage you to explore further and experiment with different datasets and configurations to unlock the full potential of AutoML on Azure.

Thank you for reading, and happy coding!

References
This guide was highly adapted from this [guide](https://github.com/Azure/azureml-examples/tree/main/sdk/python/jobs/automl-standalone-jobs/automl-nlp-text-classification-multilabel-task-paper-categorization).
