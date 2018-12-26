import azureml
from azureml.core import Workspace, Run
import urllib 


# ------- COMMAND

container_name = ""
account_name = ""
account_key = ""

dbutils.fs.mount(
source = "wasbs://"+container_name+"@"+account_name+".blob.core.windows.net",
mount_point = "/mnt/" + account_name + "/" + container_name,
extra_configs = {"fs.azure.account.key."+account_name+".blob.core.windows.net": account_key})

dbutils.fs.ls("/mnt/" + account_name + "/" + container_name)



# download data to cluster
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', filename='/tmp/train-images.gz')
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', filename='/tmp/train-labels.gz')
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', filename='/tmp/test-images.gz')
urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', filename='/tmp/test-labels.gz')