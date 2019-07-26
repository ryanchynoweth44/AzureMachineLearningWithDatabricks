# Databricks notebook source
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import datetime as dt
from pyspark.ml.feature import OneHotEncoder, VectorAssembler

# COMMAND ----------

# MAGIC %scala
# MAGIC val tags = com.databricks.logging.AttributionContext.current.tags
# MAGIC val username = tags.getOrElse(com.databricks.logging.BaseTagDefinitions.TAG_USER, java.util.UUID.randomUUID.toString.replace("-", ""))
# MAGIC spark.conf.set("com.databricks.demo.username", username)

# COMMAND ----------

client = MlflowClient() # client
exps = client.list_experiments() # get all experiments

# COMMAND ----------

exp = [s for s in exps if "/Users/{}/AzureML_MLFlow/MLFlowExp".format(spark.conf.get("com.databricks.demo.username")) in s.name][0] # get only the exp we want
exp_id = exp.experiment_id # save exp id to variable
artifact_location = exp.artifact_location # artifact location for storing
run = client.create_run(exp_id) # create the run
run_id = run.info.run_id # get the run id

# COMMAND ----------

# start and mlflow run
mlflow.start_run(run_id)

# COMMAND ----------

spark.read.format("csv").option("inferSchema", "True").option("header", "True").load("/databricks-datasets/bikeSharing/data-001/day.csv")

# COMMAND ----------

try: 
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

  # write model to datetime folder and latest folder
  lrPipelineModel.write().overwrite().save("{}/latest/bike_sharing_model.model".format(artifact_location))
  lrPipelineModel.write().overwrite().save("{}/year={}/month={}/day={}/bike_sharing_model.model".format(artifact_location, dt.datetime.utcnow().year, dt.datetime.utcnow().month, dt.datetime.utcnow().day))

  # write test predictions to datetime and lastest folder
  predictions = lrPipelineModel.transform(test_df)
  predictions.write.format("parquet").mode("overwrite").save("{}/latest/test_predictions.parquet".format(artifact_location))
  predictions.write.format("parquet").mode("overwrite").save("{}/year={}/month={}/day={}/test_predictions.parquet".format(artifact_location, dt.datetime.utcnow().year, dt.datetime.utcnow().month, dt.datetime.utcnow().day))

  # mlflow log evaluations
  evaluator = RegressionEvaluator(labelCol = "cnt", predictionCol = "prediction")

  mlflow.log_metric("mae", evaluator.evaluate(predictions, {evaluator.metricName: "mae"}))
  mlflow.log_metric("rmse", evaluator.evaluate(predictions, {evaluator.metricName: "rmse"}))
  mlflow.log_metric("r2", evaluator.evaluate(predictions, {evaluator.metricName: "r2"}))
  mlflow.set_tag("Model Path", "{}/year={}/month={}/day={}".format(artifact_location, dt.datetime.utcnow().year, dt.datetime.utcnow().month, dt.datetime.utcnow().day))

  mlflow.end_run(status="FINISHED")
  print("Model training finished successfully")
except Exception as e:
  mlflow.log_param("Error", str(e))
  mlflow.end_run(status="FAILED")
  print("Model training failed: {}".format(str(e)))

# COMMAND ----------


