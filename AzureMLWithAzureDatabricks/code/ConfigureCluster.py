import azureml
from azureml.core import Workspace, Run



# ------- COMMAND

container_name = ""
account_name = ""
account_key = ""

dbutils.fs.mount(
source = "wasbs://"+container_name+"@"+account_name+".blob.core.windows.net",
mount_point = "/mnt/" + account_name + "/" + container_name,
extra_configs = {"fs.azure.account.key."+account_name+".blob.core.windows.net": account_key})

dbutils.fs.ls("/mnt/" + account_name + "/" + container_name)


