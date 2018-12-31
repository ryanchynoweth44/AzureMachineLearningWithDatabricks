# Databricks notebook source
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import azureml
from azureml.core import Workspace, Run
from azureml.core.model import Model

# set aml workspace parameters here. 
subscription_id = ""
resource_group = ""
workspace_name = ""
workspace_region = ""

# set container and account name for data sources
container_name = ""
account_name = ""


ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)


# copy the model to local directory for deployment
model_name = "sklearn_mnist_model.pkl"
deploy_folder = os.getcwd()
dbutils.fs.cp('/dbfs/mnt/' + account_name + '/' + container_name + '/models/latest/' + model_name, "file:" + deploy_folder + "/" + model_name, True)


# register the model 
mymodel = Model.register(model_name = model_name, model_path = model_name, description = "Trained MNIST model", workspace = ws )


score = """
 
import json
import numpy as np
import os
import pickle
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

from azureml.core.model import Model
 
def init():    
  global model
  # retreive the path to the model file using the model name
  model_path = Model.get_model_path('{model_name}')
  model = joblib.load(model_path)
    
    
def run(raw_data):
  data = np.array(json.loads(raw_data)['data'])
  
  # make prediction
  y_hat = model.predict(data)
  
  # you can return any data type as long as it is JSON-serializable
  return y_hat.tolist()
    
""".format(model_name=model_name)
 
exec(score)
 
with open("score.py", "w") as file:
    file.write(score)


# Create a dependencies file
from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies.create(conda_packages=['scikit-learn']) #showing how to add libs as an eg. - not needed for this model.

with open("myenv.yml","w") as f:
    f.write(myenv.serialize_to_string())


# ACI Configuration
from azureml.core.webservice import AciWebservice, Webservice

myaci_config = AciWebservice.deploy_configuration(cpu_cores=1, 
             memory_gb=1, 
             tags={"data": "MNIST",  "method" : "sklearn"}, 
             description='Predict MNIST with sklearn')


# deploy to aci
from azureml.core.webservice import Webservice
from azureml.core.image import ContainerImage

# configure the image
image_config = ContainerImage.image_configuration(execution_script="score.py", 
                                                  runtime="python", 
                                                  conda_file="myenv.yml")

service = Webservice.deploy_from_model(workspace=ws,
                                       name='sklearn-mnist-svc',
                                       deployment_config=myaci_config,
                                       models=[mymodel],
                                       image_config=image_config)

service.wait_for_deployment(show_output=True)

# print the uri of the web service
print(service.scoring_uri)


# load compressed MNIST gz files and return numpy arrays
import gzip, struct
import numpy as np

def load_data(filename, label=False):
    with gzip.open(filename) as gz:
        struct.unpack('I', gz.read(4))
        n_items = struct.unpack('>I', gz.read(4))
        if not label:
            n_rows = struct.unpack('>I', gz.read(4))[0]
            n_cols = struct.unpack('>I', gz.read(4))[0]
            res = np.frombuffer(gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
            res = res.reshape(n_items[0], n_rows * n_cols)
        else:
            res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8)
            res = res.reshape(n_items[0], 1)
    return res


# one-hot encode a 1-D array
def one_hot_encode(array, num_of_classes):
    return np.eye(num_of_classes)[array.reshape(-1)]

X_test = load_data('/dbfs/mnt/' + account_name + '/' + container_name + '/test-images.gz', False) / 255.0
y_test = load_data('/dbfs/mnt/' + account_name + '/' + container_name + '/test-labels.gz', True).reshape(-1)


import requests
import json

# send a random row from the test set to score
random_index = np.random.randint(0, len(X_test)-1)
input_data = "{\"data\": [" + str(list(X_test[random_index])) + "]}"

headers = {'Content-Type':'application/json'}

resp = requests.post(service.scoring_uri, input_data, headers=headers)

print("POST to url", service.scoring_uri)
print("label:", y_test[random_index])
print("prediction:", resp.text)

