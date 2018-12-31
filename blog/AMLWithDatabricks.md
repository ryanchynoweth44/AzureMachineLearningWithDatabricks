# Azure Machine Learning Services and Azure Databricks

Working as a consultant in the Microsoft Azure Cloud I am continuously implementing artificial intelligent (AI) solutions in a variety of different ways to suit the client's needs. These machine learning and deep learning solutions must be easy to implement and scale as they become business critical. Most organizations have existing applications and processes that they wish to infuse AI into. Since it is often existing applications we need to deploy our intelligence so that it is easily consumable by the application. After trial and error I have grown to really love implementing solutions using the Azure Machine Learning Service (AML Service) and Azure Databricks. 

Azure Machine Learning Service is a platform that allows data scientists and data engineers to train, deploy, automate, and manage machine learning models at scale and in the cloud. Developers can build intelligent algorithms into applications and workflows using popular Python-based libraries like TensorFlow, Pytorch, and Scikit-learn. The AML Service is a framework that allows developers to train where ever they choose, then wrap their model in a docker container and deploy to any container orchestrator they wish! 

Azure Databricks is a an optimized Apache Spark Platform for heavy analytics workloads. It was designed with the founders of Apache Spark, Databricks is integrated with Azure to provide one-click setup, streamlined workflows, and an interactive workspace that enables collaboration between data scientists, data engineers, and business analysts. Developers can easily enable their business with familiar tools and a distributed processing platform to unlock their data's secrets. 

While Azure Databricks is a great platform to deploy AI Solutions, I will often use it as the compute for training machine learning models before deploying to the AML Service. 

## Ways to Implement AI
Depending on the client's need I will typically implement machine learning solutions in one of the following ways:  

- Consumable web service
- Scheduled batch process
- Continuously streaming predictions

Many organizations will start with batch processes and grow the solution to a streaming solution or a callable web service. 

### Web Service Implementation
A web service is simply code that can be invoked remotely to execute a specific task. In machine learning solutions, web services are a great way to deploy a predictive model that needs to be consumed by one or more applications. Web services allows developers to follow a microservice architecture when adding intelligence into existing or new applications.  

#### Advantages and Disadvantages
A major advantage to deploying web services over both batch and streaming solutions is the ability to add near real-time intelligence without changing infrastructure or architecture. Web services allow developers to simply add a feature to their code without having to do a massive overhaul of the current processes. 

One disadvantage is that predictions can only be made by calling the web service, therefore, if a developer wishes to have predictions made on a scheduled basis or continuously there needs to be an outside application to call that web service.  

### Batch Processing
Batch processing is a technique of transforming data at one time. Typically this is a large amount of data that has been aggregated over a period of time. The main goal of batch processing is to efficiently work on a bigger window of data that can consist of files and records. These processes are usually ran in "off" hours so that it does not impact business critical systems.  

#### Advantages and Disadvantages
Batch processing is extremely effective at unlocking deep insights in your data. It allows users to process a large window of data to analyze trends over time. 

Batch processing is extremely common for all businesses, however, there are a few disadvantages to implementing a batch process. Maintaining and debugging a batch process can sometimes be difficult. For anyone who has tried to debug a complex stored procedure in a Microsoft SQL Server will understand this difficulty. Another issue that can arise in today's cloud first world is the cost of implementing a solution. Batch solutions are great at saving money because the infrastructure required to run the process will for the most part be turned off because it only needs to be on when the process is running. However, the implementation and knowledge transfer of the solution can often be the first hurdle faced. 

By thoughtfully designing and documenting these batch processes, organizations should be able to avoid any issues with these types of solutions.  

### Stream Processing
Stream processing is the ability to analyze data as it flows from the data source (application, devices etc.) to a storage location (relational databases, data lakes etc.). Due to the continuous nature of these systems, large amounts of data is not required to be stored at one time and are focused on finding insights in small windows of time. Stream processing is ideal when you wish to track or detect events that are close in time and occur frequently. 

#### Advantages and Disadvantages
The hardest part of implementing a streaming data solution is the ability to keep up with the input data rate. Meaning that the solution must be able to process data as fast or faster than the rate at which the data sources generate data. If the solution is unable to achieve this then it will lead to a never ending backlog of data and may run into storage or memory issues. 

In addition to processing speed planning to operate within the system. Having a plan to access data after the stream is operated on and reduce the number of copies to optimize storage can be difficult.  

## Check out the Walkthrough
Implementing a machine learning solution with Azure Databricks and Azure Machine Learning allows data scientists to easily deploy the same model in several different environments. Azure Databricks is capable of making streaming predictions as data enters the system, as well as large batch processes. While these two ways are great for unlocking insights from your data, often the best way to incorporate intelligence into an application is by calling a web service. Azure Machine Learning service allows a data scientist to wrap up their model and easily deploy it to Azure Container Instance. From my experience this is the best and easiest why to integrate intelligence into existing applications and processes!  

Check out the [walkthrough](https://github.com/ryanchynoweth44/AzureMachineLearningWithDatabricks) I created that shows developers how to train a model on the Databricks platform and deploys that model to AML Service.  


