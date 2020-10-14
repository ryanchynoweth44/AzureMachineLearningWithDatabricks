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


1. Once the above steps are complete, we will want to set up our development environment. This includes entering secrets into our Azure Key Vault. For reference use the [create_secrets.py](../code/create_secrets.py) script. Please supply the appropriate variables below and notice that we save our configuration to a local config file.  
    ```python
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
    ```

1. Next we will create a compute cluster using our [create_cluster.py](../code/create_cluster.py) script. This process creates a remote compute environment that allows us to launch individual experiment runs on different compute engines. The remote compute feature is formally known as Azure Batch AI. Notice in the script we use the configuration file from the previous step to connect and obtain our secrets.  
    ```python
    import os 
    from azureml.core import Workspace
    from azureml.core.compute import ComputeTarget, AmlCompute
    from azureml.core.compute_target import ComputeTargetException
    from azureml.core.authentication import ServicePrincipalAuthentication

    ws = Workspace.from_config()
    kv = ws.get_default_keyvault()

    workspace_name = ws.name
    subscription_id = kv.get_secret('subscriptionId')
    resource_group = kv.get_secret('resourceGroup')
    tenant_id = kv.get_secret('tenantId')
    client_id = kv.get_secret('clientId')
    client_secret = kv.get_secret('clientSecret')
    print("Azure ML SDK Version: ", azureml.core.VERSION)


    # connect to your aml workspace
    ## NOTE: you can use Workspace.create to create a workspace using Python.  
    ## this authentication method will require a 
    auth = ServicePrincipalAuthentication(tenant_id=tenant_id, service_principal_id=client_id, service_principal_password=client_secret)
    ws = Workspace.get(name=workspace_name, auth=auth, subscription_id=subscription_id, resource_group=resource_group)


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