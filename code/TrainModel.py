import azureml
from azureml.core import Workspace, Run, Experiment
import numpy as np
# check core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)


# set aml workspace parameters here. 
subscription_id = "<your-subscription-id>"
resource_group = "<your-resource-group>"
workspace_name = "<your-workspace-name>"
workspace_region = "<your-region>"


ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)