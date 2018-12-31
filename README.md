# Azure Machine Learning With Azure Databricks
This repository highlights the workflow and ease of use to train machine learning or deep learning models using Azure Databricks. Then deploying those models on both the AML Service and Azure Databricks. There are many demos and documentation of this workflow available by Microsoft, however, I hope to provide and end to end walkthrough in a single location.   

We will be using the popular MNIST dataset, and will be following closely with this Azure Machine Learning Service example of [training a model](https://github.com/Azure/MachineLearningNotebooks/blob/master/tutorials/img-classification-part1-training.ipynb). The walkthrough also teaches users how to [deploy models](https://github.com/Azure/MachineLearningNotebooks/blob/fb6a73a7906bcde374887c8fafbce7ae290db435/tutorials/img-classification-part2-deploy.ipynb) using the Azure Machine Learning service. The linked example is ran using Azure Notebooks, which is an excellent way to use cloud compute resources while staying in a Jupyter environment. The key difference here is that we will be using Azure Databricks to train and deploy our model.   

## Blog
This type of model training and deployment is common with clients I work with. Therefore, I wrote a [blog](./blog/AMLWithDatabricks.md) describing tips and why a developer would choose to deploy using AML Service and Databricks.  

## Walkthrough
Please complete the following in order for an end to end implementation:  
1. [Set up your Azure environment](./walkthrough/01_SetUpAzureEnvironment)
1. [Train a machine learning model](./walkthrough/02_TrainModel.md)
1. [Deploy model to AML Service](./walkthrough/03_DeployModel.md)


### Automation
The walkthrough shows how to do manual deployments of models using AML Service, however, one thing to keep in mind is that typically these would be automated with build and release pipelines ([Azure DevOps](https://azure.microsoft.com/en-us/services/devops/)). If a data scientist wishes to deploy the model to a web service environment as shown in the walkthrough I would recommend the pipelines access the models directly from the model management azure machine learning workspace, while if a streaming solution is desired I would access the model directly from a databricks file system mount. 

## Conclusion
This repo aims to provide an overview of both Azure Databricks and Azure Machine Learning Services. If there are any confusing steps or errors please let me know. Any other comments or questions you can contact me at rchynoweth@10thmagnitude.com. 