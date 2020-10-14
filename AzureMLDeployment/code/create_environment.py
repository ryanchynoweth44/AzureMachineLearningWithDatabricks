from azureml.core import Workspace, Environment


ws = Workspace.from_config()
kv = ws.get_default_keyvault()

exp_name = 'remote_training'
workspace_name = ws.name
subscription_id = kv.get_secret('subscriptionId')
resource_group = kv.get_secret('resourceGroup')
tenant_id = kv.get_secret('tenantId')
client_id = kv.get_secret('clientId')
client_secret = kv.get_secret('clientSecret')
print("Azure ML SDK Version: ", azureml.core.VERSION)


env = Environment.from_conda_specification(name='sklearn-env', file_path='.azureml/env.yml')
env.register(ws) # register the environment!