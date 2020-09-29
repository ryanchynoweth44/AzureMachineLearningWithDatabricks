import azureml.core
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Run
from azureml.core.authentication import ServicePrincipalAuthentication


exp_name = 'local_training'
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
exp = Experiment(ws, exp_name)

config = ScriptRunConfig(source_directory='AzureMLDeployment/code', script='train.py', compute_target='local')

env = Environment.from_conda_specification(name='sklearn-env', file_path='AzureMLDeployment/code/env.yml')

config.run_config.environment = env

run = exp.submit(config)
print(run.get_portal_url())

r = Run(exp, run.id)
r.get_details()