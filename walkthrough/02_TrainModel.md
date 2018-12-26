# Train an MNIST Model in Azure Databricks

## Train Model
1. Import the required libraries 
    ```python 
    import azureml
    from azureml.core import Workspace, Run, Experiment
    import numpy as np
    # check core SDK version number
    print("Azure ML SDK Version: ", azureml.core.VERSION)
    ```

1. Next we need to connect to our Azure Machine Learning Workspace. First set the variables below. Please note that you can use the [Databricks Secrets API](https://docs.databricks.com/api/latest/secrets.html) to securely handle secrets in production environments.   

    ```python 
    # set aml workspace parameters here. 
    subscription_id = "<your-subscription-id>"
    resource_group = "<your-resource-group>"
    workspace_name = "<your-workspace-name>"
    workspace_region = "<your-region>"
    ```

1. Next connect to your workspace. Please note that you may need to sign in interactively using a browser.  
    ```python
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
    ```

1. Create an experiment to track the runs in your workspace. A workspace can have muliple experiments.
    ```python 
    experiment_name = 'sklearn-mnist'

    exp = Experiment(workspace=ws, name=experiment_name)
    ```
