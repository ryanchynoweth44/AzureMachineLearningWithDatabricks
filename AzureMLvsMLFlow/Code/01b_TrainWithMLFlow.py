# Databricks notebook source
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import datetime as dt
from pyspark.ml.feature import OneHotEncoder, VectorAssembler

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder 

# COMMAND ----------

spark.conf.set("spark.databricks.mlflow.trackMLlib.enabled", "true")

# COMMAND ----------

# MAGIC %scala
# MAGIC val tags = com.databricks.logging.AttributionContext.current.tags
# MAGIC val username = tags.getOrElse(com.databricks.logging.BaseTagDefinitions.TAG_USER, java.util.UUID.randomUUID.toString.replace("-", ""))
# MAGIC spark.conf.set("com.databricks.demo.username", username)

# COMMAND ----------

client = MlflowClient() # client
exps = client.list_experiments() # get all experiments

# COMMAND ----------

exps

# COMMAND ----------

exp = [s for s in exps if "/Users/{}/exps/MLFlowExp".format(spark.conf.get("com.databricks.demo.username")) in s.name][0] # get only the exp we want
exp_id = exp.experiment_id # save exp id to variable
artifact_location = exp.artifact_location # artifact location for storing
run = client.create_run(exp_id) # create the run
run_id = run.info.run_id # get the run id

# COMMAND ----------

# start and mlflow run
mlflow.start_run(run_id)

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

# COMMAND ----------

# One Hot Encoding
mnth_encoder = OneHotEncoder(inputCol="mnth", outputCol="encoded_mnth")
weekday_encoder = OneHotEncoder(inputCol="weekday", outputCol="encoded_weekday")

# set the training variables we want to use
train_cols = ['encoded_mnth', 'encoded_weekday', 'temp', 'hum']

# convert cols to a single features col
assembler = VectorAssembler(inputCols=train_cols, outputCol="features")


# COMMAND ----------

dt = DecisionTreeRegressor(featuresCol="features", labelCol="cnt")

# COMMAND ----------

# Create pipeline
pipeline = Pipeline(stages=[
    mnth_encoder,
    weekday_encoder,
    assembler,
    dt
])

# COMMAND ----------

grid = (ParamGridBuilder()
  .addGrid(dt.maxDepth, [2, 3, 4, 5, 6, 7, 8])
  .addGrid(dt.maxBins, [2, 4, 8])
  .build())

# COMMAND ----------

valid_eval =  RegressionEvaluator(labelCol = "cnt", predictionCol = "prediction", metricName="rmse")

# COMMAND ----------

cv = CrossValidator(estimator=pipeline, evaluator=valid_eval, estimatorParamMaps=grid, numFolds=3)

# COMMAND ----------

cvModel = cv.fit(train_df)
mlflow.set_tag('owner_team', spark.conf.get("com.databricks.demo.username")) # Logs user-defined tags
test_metric = valid_eval.evaluate(cvModel.transform(test_df))
mlflow.log_metric('test_' + valid_eval.getMetricName(), test_metric) # Logs additional metrics


# COMMAND ----------

bestModel = cvModel.bestModel
# write test predictions to datetime and lastest folder
predictions = bestModel.transform(test_df)
# mlflow log evaluations
evaluator = RegressionEvaluator(labelCol = "cnt", predictionCol = "prediction")

mlflow.log_metric("mae", evaluator.evaluate(predictions, {evaluator.metricName: "mae"}))
mlflow.log_metric("rmse", evaluator.evaluate(predictions, {evaluator.metricName: "rmse"}))
mlflow.log_metric("r2", evaluator.evaluate(predictions, {evaluator.metricName: "r2"}))

# COMMAND ----------

lrPipelineModel.write().overwrite().save("{}/latest/bike_sharing_model.model".format(artifact_location))
lrPipelineModel.write().overwrite().save("{}/year={}/month={}/day={}/bike_sharing_model.model".format(artifact_location, dt.datetime.utcnow().year, dt.datetime.utcnow().month, dt.datetime.utcnow().day))

# write test predictions to datetime and lastest folder

predictions.write.format("parquet").mode("overwrite").save("{}/latest/test_predictions.parquet".format(artifact_location))
predictions.write.format("parquet").mode("overwrite").save("{}/year={}/month={}/day={}/test_predictions.parquet".format(artifact_location, dt.datetime.utcnow().year, dt.datetime.utcnow().month, dt.datetime.utcnow().day))
mlflow.set_tag("Model Path", "{}/year={}/month={}/day={}".format(artifact_location, dt.datetime.utcnow().year, dt.datetime.utcnow().month, dt.datetime.utcnow().day))

# COMMAND ----------

mlflow.end_run(status="FINISHED")
