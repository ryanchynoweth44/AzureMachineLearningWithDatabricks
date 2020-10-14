import os 
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

workspace_name = ""
subscription_id = ""
resource_group = ""
tenant_id = ""
client_id = ""
client_secret = ""


auth = ServicePrincipalAuthentication(tenant_id=tenant_id, service_principal_id=client_id, service_principal_password=client_secret)
ws = Workspace.get(name=workspace_name, auth=auth, subscription_id=subscription_id, resource_group=resource_group)

kv = ws.get_default_keyvault()

kv.set_secret('workspaceName',workspace_name)
kv.set_secret('subscriptionId',subscription_id)
kv.set_secret('resourceGroup',resource_group)
kv.set_secret('tenantId',tenant_id)
kv.set_secret('clientId',client_id)
kv.set_secret('clientSecret',client_secret)

ws.write_config()



