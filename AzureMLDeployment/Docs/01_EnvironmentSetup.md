# Environment Setup

All code for this demo is in the [code](code) directory, but all commands should be executed from the repository root. To get started make sure the following is installed on your machine. 
- VS Code
- Anaconda


1. Use the Azure Portal to [create an Azure Machine Learning Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace).   
    - This should result in the following resources:
        - Azure Machine Learning Workspace
        - Storage Account
        - Key Vault
        - Application Insights

1. [Create a service principal](https://docs.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal) and grant it access to the azure machine learning workspace resource. We will be using this for authentication.  


1. Open up an anaconda command prompt and create a Python 3.7 environment.    
    ```
    conda create -n amlenv python=3.6 -y

    conda activate amlenv
    ```
    **NOTE** - I originally created a Python 3.7 environment and I was unable to install the `azureml-sdk`. The [pypi](https://pypi.org/project/azureml-sdk/) page states that it works for python 3.5, 3.6, 3.7, and 3.8 but I found that it may not be the case.

1. Pip install all libraries in the [`requirements.txt`](code/requirements.txt) file.  
    ```
    azureml-sdk==1.14.0
    scikit-learn==0.23.2
    pandas==1.1.2
    seaborn==0.11.0
    ```


1. Once the above steps are complete, we will create a compute cluster in for remote training of experiments. The following steps can be found in the [create_cluster.py](../code/create_cluster.py) script. 
    ```python
    # import libraries
    import os 
    from azureml.core import Workspace
    from azureml.core.compute import ComputeTarget, AmlCompute
    from azureml.core.compute_target import ComputeTargetException
    from azureml.core.authentication import ServicePrincipalAuthentication
    ``` 

1. Set environment variables. You can hardcode these or use the os environment variables like I did. 
    ```python
    workspace_name = os.environ.get('workspace_name')
    subscription_id = os.environ.get('subscription_id')
    resource_group = os.environ.get('resource_group')
    tenant_id = os.environ.get('tenant_id')
    client_id = os.environ.get('client_id')
    client_secret = os.environ.get('client_secret')
    print("Azure ML SDK Version: ", azureml.core.VERSION)
    ```

1. Connect to your workspace using the service principal. 
    ```python
    # connect to your aml workspace
    ## NOTE: you can use Workspace.create to create a workspace using Python.  
    ## this authentication method will require a 
    auth = ServicePrincipalAuthentication(tenant_id=tenant_id, service_principal_id=client_id, service_principal_password=client_secret)
    ws = Workspace.get(name=workspace_name, auth=auth, subscription_id=subscription_id, resource_group=resource_group)
    ```

1. Create your cluster. In our case we are deploying a 0-4 node cluster of STANDARD_D2_V2 machines that will auto-terminate after 30 minutes. Please note that these clusters allow us to launch multiple experiements in parallel and do not allow for distributed machine learning. The parallel training is most commonly used during hyperparameter tuning.  
    ```python
    # Choose a name for your CPU cluster
    cpu_cluster_name = "cpu-cluster"

    # Verify that the cluster does not exist already
    try:
        cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
        print('Found existing cluster, use it.')
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                                max_nodes=4, 
                                                                idle_seconds_before_scaledown=1800)
        cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

    cpu_cluster.wait_for_completion(show_output=True)
    ```