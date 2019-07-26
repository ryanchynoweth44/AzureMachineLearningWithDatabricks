# Databricks notebook source
# MLFlow Notebook
import mlflow
from mlflow.tracking import MlflowClient
import azureml
from azureml.core import Workspace, Run, Experiment

import os, shutil

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import datetime as dt
from pyspark.ml.feature import OneHotEncoder, VectorAssembler

# COMMAND ----------

# set aml workspace parameters here. 
subscription_id = ""
resource_group = ""
workspace_name = ""
workspace_region = ""

ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)

# COMMAND ----------

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

# COMMAND ----------

# create experiment
experiment_name = 'bikeSharingDemandMLFlowAML'
exp = Experiment(workspace=ws, name=experiment_name)

# COMMAND ----------

mlflow.set_experiment(experiment_name)

run = mlflow.start_run()
run_id = run.info.run_uuid
exp_id = run.info.experiment_id
artifact_location = run.info.artifact_uri

# COMMAND ----------

azRun = Run(exp, run_id)

# COMMAND ----------

df = (spark
      .read
      .format("csv")
      .option("inferSchema", "True")
      .option("header", "True")
      .load("/databricks-datasets/bikeSharing/data-001/day.csv")
     )
# split data
train_df, test_df = df.randomSplit([0.7, 0.3])

# One Hot Encoding
mnth_encoder = OneHotEncoder(inputCol="mnth", outputCol="encoded_mnth")
weekday_encoder = OneHotEncoder(inputCol="weekday", outputCol="encoded_weekday")

# set the training variables we want to use
train_cols = ['encoded_mnth', 'encoded_weekday', 'temp', 'hum']

# convert cols to a single features col
assembler = VectorAssembler(inputCols=train_cols, outputCol="features")

# Set linear regression model
lr = LinearRegression(featuresCol="features", labelCol="cnt")

# Create pipeline
pipeline = Pipeline(stages=[
    mnth_encoder,
    weekday_encoder,
    assembler,
    lr
])

# fit pipeline
lrPipelineModel = pipeline.fit(train_df)

# write test predictions to datetime and lastest folder
predictions = lrPipelineModel.transform(test_df)
# mlflow log evaluations
evaluator = RegressionEvaluator(labelCol = "cnt", predictionCol = "prediction")

mlflow.log_metric("mae", evaluator.evaluate(predictions, {evaluator.metricName: "mae"}))
mlflow.log_metric("rmse", evaluator.evaluate(predictions, {evaluator.metricName: "rmse"}))
mlflow.log_metric("r2", evaluator.evaluate(predictions, {evaluator.metricName: "r2"}))

# COMMAND ----------

model_nm = "bikeshare.mml"
model_output = '/mnt/azml/outputs/'+model_nm
model_dbfs = "/dbfs"+model_output
lrPipelineModel.write().overwrite().save(model_output)

# COMMAND ----------

model_name, model_ext = model_dbfs.split(".")

# COMMAND ----------

model_zip = model_name + ".zip"
shutil.make_archive(model_name, 'zip', model_dbfs)
azRun.upload_file("outputs/" + model_nm, model_zip)

# COMMAND ----------

azRun.register_model(model_name = 'model_nm', model_path = "outputs/" + model_nm)

# COMMAND ----------

# now delete the serialized model from local folder since it is already uploaded to run history 
shutil.rmtree(model_dbfs)
os.remove(model_zip)

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------


