# Deploying a model to Azure Container Instances

In this part of the example we will deploy our model as a web service hosted as a docker container in Azure Container Instances. 

### Create a Score Script

There are two required functions to build a web service in Azure Machine Learning: `init` and `run`.  The `init` function is loaded upon the deployment of the service and is usually intended to load variables like the model or other required functions.  The `run` function is executed each time the service is called. It is possible to have more functions available for data transformations or saving predictions to a database but we will not do that here.  


Here is our score script: 
```python
import json
import numpy as np
import os
import pickle
import logging


def init():
    global mod 
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'decision_tree_model.pkl')
    logging.log(20, "Model Path: {}".format(model_path))
    with open(model_path, 'rb') as f:
        mod = pickle.load(f)
    logging.log(20, "Model Loaded. ")


def run(data):
    data = np.array(json.loads(data)['data'])
    logging.log(20, data)
    # make prediction
    y_hat = mod.predict(data)
    logging.log(20, y_hat)
    # you can return any data type as long as it is JSON-serializable
    return y_hat.tolist()
```

### Create a deployment script 

Our deployment script will configure the container instance and package our scoring script into a web service. You will notice that we have `auth_enabled` set to `True` for our deployment. This will allow us to use a Bearer token for authentication against the service. If we were deploying to AKS we would use [`AksEndpointDeploymentConfiguration`](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.webservice.aks.aksendpointdeploymentconfiguration?view=azure-ml-py) configuration and could either use the key authentication as seen in this example, or we could use token authentication which would first authenticate against Azure Active Directory before authenticating against the web service.  

```python
import random
import requests
import json
import pandas as pd
import azureml.core
from azureml.core.webservice import AciWebservice
from azureml.core.webservice import Webservice
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core.keyvault import Keyvault
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.authentication import ServicePrincipalAuthentication

# Display the core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)

workspace_name = os.environ.get('workspace_name')
subscription_id = os.environ.get('subscription_id')
resource_group = os.environ.get('resource_group')
tenant_id = os.environ.get('tenant_id')
client_id = os.environ.get('client_id')
client_secret = os.environ.get('client_secret')
print("Azure ML SDK Version: ", azureml.core.VERSION)

# connect to your aml workspace
## NOTE: you can use Workspace.create to create a workspace using Python.  
## this authentication method will require a 
auth = ServicePrincipalAuthentication(tenant_id=tenant_id, service_principal_id=client_id, service_principal_password=client_secret)
ws = Workspace.get(name=workspace_name, auth=auth, subscription_id=subscription_id, resource_group=resource_group)


aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
    memory_gb=1, 
    tags={"data": "Titanic",  "method" : "sklearn"}, 
    description='Predict Titanic with sklearn',
    auth_enabled=True
    )



model = Model(ws, 'decision_tree_model')


env = Environment.from_conda_specification(name='sklearn-env', file_path='.azureml/env.yml')
inference_config = InferenceConfig(entry_script="src/score.py", environment=env)

service = Model.deploy(workspace=ws, 
    name='sklearn-titanic', 
    models=[model], 
    inference_config=inference_config, 
    deployment_config=aciconfig,
    overwrite=True)

service.wait_for_deployment(show_output=True)

```




### Test the Service 

Using the same script above we can test our web service. You will notice that we preprocess all data before sending it to the API. There are a few different options for data preprocessing: 
1. Preprocess before sending the request (which is what we do here). 
1. Leverage pipelines to execute preprocessing by a different service (next example)
1. Preprocess data within the API (not shown in this repo). 
    1. Essentially we would add another method to our `score.py` file that would process the data when the API receives it. This is usually ideal for small transformations but machine learning often requires a lot of transformations for a machine learning dataset (data scaling, aggregation, query additional data etc.) so a pipeline is usually the best option. 

First we will load our data.  
```python
df = pd.read_csv("https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")
df = pd.get_dummies(df,prefix=['Pclass'], columns = ['Pclass'])
df = pd.get_dummies(df,prefix=['Sex'], columns = ['Sex'])
df = df.dropna()
df.head()

# identify a handful of training columns
label = 'Survived'
training_cols = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_male', 'Sex_female', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard']

item = df[training_cols].values[random.randint(0, len(df))]
```

Next we can use the Azure ML SDK to send a request to the API.  
```python
input_data = "{\"data\": [" + str(list(item)) + "]}"
y_hat = service.run(input_data=input_data)
y_hat
```
Output:  
`>> [1]`

Our we can use the `requests` library to complete an HTTP call.  
```python
input_data = "{\"data\": [" + str(list(item)) + "]}"

headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer '+service.get_keys()[0]}

resp = requests.post(service.scoring_uri, input_data, headers=headers)
resp.content.decode('utf-8')
```
Output:  
`>> '[1]'`