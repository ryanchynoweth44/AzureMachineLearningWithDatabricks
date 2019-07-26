# Databricks notebook source
container_name = ""
account_name = ""
account_key = ""

# COMMAND ----------

dbutils.fs.mount(
    source = "wasbs://{}@{}.blob.core.windows.net".format(container_name, account_name),
    mount_point = "/mnt/{}/{}".format(account_name, container_name),
    extra_configs = {"fs.azure.account.key."+account_name+".blob.core.windows.net": account_key})

# COMMAND ----------


