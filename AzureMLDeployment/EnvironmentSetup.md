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
    ```


