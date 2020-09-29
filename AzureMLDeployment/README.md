# Azure Machine Learning End-to-End Example

To follow this example all code should be executed from the AzureMLDeployment directory. Steps are provided in the [Docs](Docs) folder in the following order:  
1. [Environment Setup](Docs/01_EnvironmentSetup.md) 
1. [Train a model](Docs/02_TrainModel.md)
    - In this step we will train a model locally and on a remote virtual machine. In my opinion the remote compute targets are a major benefit to machine learning and prefered over other targets i.e. Databricks, because if I wanted to use Databricks for training I would write my code within the Databricks environment and follow one of the examples above. 
1. [Deploy Model](Docs/03_DeployModel.md)
1. [Create a pipeline](Docs/04_CreatePipeline.md)
1. [Redeploy Model and Pipeline](Docs/04_CreatePipeline.md)


Not covered in this example is TLS/SSL encryption, please reference the following [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-secure-web-service). In this tutorial we deploy a publically exposed API (which can be restricted to specific IP addresses) that is secured with an API key. All traffic is HTTP not HTTPS.   


## Result 

This walkthrough showed the process of training a machine learning model and deploying it as a web service to be called via REST API calls. It is important to note that this process of training is easily reproducible making the scheduled retraining of a service simple. We have all the scripts that can be automated to start/stop a virtual machine, train a model, log metadata information, register the model, and deploy the model as a service. Furthermore, we can leverage our compute and train a model using cross-validation to always ensure the optimal hyperparameters are tuned for each deployment as new data is added to the training set. 

