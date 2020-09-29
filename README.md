# Azure Machine Learning With Azure Databricks

The original purpose of this repository is to highlight the workflow and ease of use to train machine learning or deep learning models using Azure Databricks and Azure Machine Learning Service, however, it is evolving into general examples of both services. Therefore, some examples provide may include just Azure Databricks or Azure Machine Learning.  

PLEASE NOTE THAT THIS REPOSITORY IS NOT ACTIVELY MAINTAINED, BUT WILL BE UPDATED AS ISSUES ARE CREATED.  


## Examples

**Azure Machine Learning with Azure Databricks**  
We will be using the popular MNIST dataset, and will be following closely with this Azure Machine Learning Service example of [training a model](https://github.com/Azure/MachineLearningNotebooks/blob/master/tutorials/img-classification-part1-training.ipynb). The walkthrough also teaches users how to [deploy models](https://github.com/Azure/MachineLearningNotebooks/blob/fb6a73a7906bcde374887c8fafbce7ae290db435/tutorials/img-classification-part2-deploy.ipynb) using the Azure Machine Learning service. The linked example is ran using Azure Notebooks, which is an excellent way to use cloud compute resources while staying in a Jupyter environment. The key difference here is that we will be using Azure Databricks to train and deploy our model.   
Please complete the following in order for an end to end implementation:  
1. [Set up your Azure environment](./AzureMLWithAzureDatabricks/walkthrough/01_SetUpAzureEnvironment.md)
1. [Train a machine learning model](./AzureMLWithAzureDatabricks/walkthrough/02_TrainModel.md)
1. [Deploy model to AML Service](./AzureMLWithAzureDatabricks/walkthrough/03_DeployModel.md)

**Azure Machine Learning vs MLFlow**  
Please complete the following in order for an end to end implementation:  
1. [Set up your Azure environment](./AzureMLvsMLFlow/Docs/00_SetUpAzureEnvironment.md)
1. [Train a machine learning model with MLFlow](./AzureMLvsMLFlow/Docs/01_TrainWithMLFlow.md)
1. [Train a machine learning model with AzureML](./AzureMLvsMLFlow/Docs/02_TrainWithAzureML.md)
1. [Cross Validation with MLFlow](./AzureMLvsMLFlow/Docs/03_CrossValidation.md)
1. [Cross Validation with MLFlow and Azure ML](./AzureMLvsMLFlow/Docs/04_TrainWithBoth.md)


**Azure Machine Learning with VS Code and Anaconda**
After over a year of not using Azure Machine Learning, I discovered that there has been a number of updates to where it is worth my time to explore an example training and deployment process. After going through this example, it is clear that much of the behind the scenes resources and workflow are the same but small details have been changed with new releases. For this example, we will be using the titanic dataset. Please complete in the following order: 
1. [Environment Setup](AzureMLDeployment/Docs/01_EnvironmentSetup.md) 
1. [Train a model](AzureMLDeployment/Docs/02_TrainModel.md)
    - In this step we will train a model locally and on a remote virtual machine. In my opinion the remote compute targets are a major benefit to machine learning and prefered over other targets i.e. Databricks, because if I wanted to use Databricks for training I would write my code within the Databricks environment and follow one of the examples above. 
1. [Deploy Model](AzureMLDeployment/Docs/03_DeployModel.md)
1. [Create a pipeline](AzureMLDeployment/Docs/04_CreatePipeline.md)
1. [Redeploy Model and Pipeline](AzureMLDeployment/Docs/04_CreatePipeline.md)


## Conclusion
This repo aims to provide an overview of both Azure Databricks and Azure Machine Learning Services. If there are any confusing steps or errors please let me know. Any other comments or questions you can contact me at ryanachynoweth@gmail.com.
