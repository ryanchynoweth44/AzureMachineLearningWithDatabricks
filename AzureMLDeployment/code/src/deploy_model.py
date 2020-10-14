import random
import uuid
import requests
import json
import pandas as pd
import azureml.core
from azureml.core.webservice import AciWebservice
from azureml.core.webservice import Webservice
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core.keyvault import Keyvault
from azureml.core import Workspace, Run
from azureml.core.model import Model
from azureml.core.authentication import ServicePrincipalAuthentication

run = Run.get_context()

exp_name = 'titanic_api_deployment'
workspace_name = run.get_secret('workspaceName')
subscription_id = run.get_secret('subscriptionId')
resource_group = run.get_secret('resourceGroup')
tenant_id = run.get_secret('tenantId')
client_id = run.get_secret('clientId')
client_secret = run.get_secret('clientSecret')
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


env = Environment.get(workspace=ws, name="sklearn-env")
inference_config = InferenceConfig(entry_script="src/score.py", environment=env)

service = Model.deploy(workspace=ws, 
    name='sklearn-titanic', 
    models=[model], 
    inference_config=inference_config, 
    deployment_config=aciconfig,
    overwrite=True)

service.wait_for_deployment(show_output=True)



######### Lets test the web service 

df = pd.read_csv("https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")
df = pd.get_dummies(df,prefix=['Pclass'], columns = ['Pclass'])
df = pd.get_dummies(df,prefix=['Sex'], columns = ['Sex'])
df = df.dropna()
df.head()

# identify a handful of training columns
label = 'Survived'
training_cols = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_male', 'Sex_female', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard']

item = df[training_cols].values[random.randint(0, len(df))]

#### using the sdk #### 

input_data = "{\"data\": [" + str(list(item)) + "]}"
y_hat = service.run(input_data=input_data)
y_hat

#### using http ####


# send a random row from the test set to score
input_data = "{\"data\": [" + str(list(item)) + "]}"

headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer '+service.get_keys()[0]}

# for AKS deployment you'd need to the service key in the header as well
# api_key = service.get_key()
# headers = {'Content-Type':'application/json',  'Authorization':('Bearer '+ api_key)} 

resp = requests.post(service.scoring_uri, input_data, headers=headers)
resp.content.decode('utf-8')

