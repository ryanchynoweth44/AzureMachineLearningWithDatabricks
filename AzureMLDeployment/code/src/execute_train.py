import os
import azureml.core
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Run
from azureml.core.authentication import ServicePrincipalAuthentication

os.makedirs('outputs', exist_ok=True)

run = Run.get_context()

exp_name = 'titanic_training'
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

exp = Experiment(ws, exp_name)

env = Environment.get(workspace=ws, name="sklearn-env")
config = ScriptRunConfig(source_directory='src', script='train.py', compute_target='cpu-cluster', environment=env)


run = exp.submit(config)
print(run.get_portal_url())

r = Run(exp, run.id)
r.get_details()