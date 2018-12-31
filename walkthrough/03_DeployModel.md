# Deploy a MNIST Model using Azure Machine Learning service 
In this step we will walkthrough how to take the model we trained using Azure Databricks, and deploy it to Azure Container Instance.  

Please note that this walkthrough follows similarly to this [demo](https://github.com/Azure/MachineLearningNotebooks/blob/fb6a73a7906bcde374887c8fafbce7ae290db435/tutorials/img-classification-part2-deploy.ipynb) by Microsoft. 

## Setting up the environment
1. Create a new python notebook called "DeployModel". 

1. Import the require python libraries.  
    ```python
    import os
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    
    import azureml
    from azureml.core import Workspace, Run
    from azureml.core.model import Model
    ```

1. Next we need to connect to our Azure Machine Learning Workspace. First set the variables below. Please note that you can use the [Databricks Secrets API](https://docs.databricks.com/api/latest/secrets.html) to securely handle secrets in production environments.   

    ```python 
    # set aml workspace parameters here. 
    subscription_id = "<your-subscription-id>"
    resource_group = "<your-resource-group>"
    workspace_name = "<your-workspace-name>"
    workspace_region = "<your-region>"

    # set container and account name for data sources
    container_name = ""
    account_name = ""
    ```

1. Next connect to your workspace. Please note that you may need to sign in interactively using a browser.  
    ```python
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
    ```

1. The Azure Machine Learning service expects the model to be in the working directory. Therefore, we will copy it from the mounted storage account and download it locally to our cluster.   
    ```python
    # copy the model to local directory for deployment
    model_name = "sklearn_mnist_model.pkl"
    deploy_folder = os.getcwd()
    dbutils.fs.cp('/dbfs/mnt/' + account_name + '/' + container_name + '/models/latest/' + model_name, "file:" + deploy_folder + "/" + model_name, True)
    ```

1. Since we are deploying a model we will want to first register it with our AML Service workspace.  
    ```python
    # register the model 
    mymodel = Model.register(model_name = model_name, model_path = model_name, description = "Trained MNIST model", workspace = ws )
    ```

1. Next we need to write a scoring file to our cluster as well. This is the code that will execute when the web service is called.  
    ```python
    #%%writefile score.py
    score = """
    import json
    import numpy as np
    import os
    import pickle
    from sklearn.externals import joblib
    from sklearn.linear_model import LogisticRegression

    from azureml.core.model import Model
    
    def init():    
    global model
    # retreive the path to the model file using the model name
    model_path = Model.get_model_path('{model_name}')
    model = joblib.load(model_path)
        
        
    def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    
    # make prediction
    y_hat = model.predict(data)
    
    # you can return any data type as long as it is JSON-serializable
    return y_hat.tolist()
        
    """.format(model_name=model_name)
    
    exec(score)
    
    with open("score.py", "w") as file:
        file.write(score)
    ```

1. Next we need to create our config file for deployment. 
    ```python
    # Create a dependencies file
    from azureml.core.conda_dependencies import CondaDependencies 

    myenv = CondaDependencies.create(conda_packages=['scikit-learn']) #showing how to add libs as an eg. - not needed for this model.

    with open("myenv.yml","w") as f:
        f.write(myenv.serialize_to_string())
    ```

1. We will now configure an Azure Container Instance to deploy to. This will be deployed to our Azure Machine Learning Service Workspace.  
    ```python
    # ACI Configuration
    from azureml.core.webservice import AciWebservice, Webservice

    myaci_config = AciWebservice.deploy_configuration(cpu_cores=1, 
                memory_gb=1, 
                tags={"data": "MNIST",  "method" : "sklearn"}, 
                description='Predict MNIST with sklearn')
    ```

1. Now we are ready to actually deploy our model as a web service.  
    ```python
    # deploy to aci
    from azureml.core.webservice import Webservice
    from azureml.core.image import ContainerImage

    # configure the image
    image_config = ContainerImage.image_configuration(execution_script="score.py", 
                                                    runtime="python", 
                                                    conda_file="myenv.yml")

    service = Webservice.deploy_from_model(workspace=ws,
                                        name='sklearn-mnist-svc',
                                        deployment_config=myaci_config,
                                        models=[mymodel],
                                        image_config=image_config)

    service.wait_for_deployment(show_output=True)
    ```

1. Once the image is deployed you can run the following commands to get service logs. This is most useful when the deployment fails. 
    ```python 
    # if you already have the service object handy
    print(service.get_logs())

    # if you know the service name
    print(ws.webservices()['<service name here>'].get_logs())
    ```

1. You can print the url of the web service if you wish.  
    ```python
    print(service.scoring_uri)
    ```

1. For testing purposes run the following command to get test data. 
    ```python
    import gzip
    import numpy as np
    import struct
    # load compressed MNIST gz files and return numpy arrays
    def load_data(filename, label=False):
        with gzip.open(filename) as gz:
            struct.unpack('I', gz.read(4))
            n_items = struct.unpack('>I', gz.read(4))
            if not label:
                n_rows = struct.unpack('>I', gz.read(4))[0]
                n_cols = struct.unpack('>I', gz.read(4))[0]
                res = np.frombuffer(gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
                res = res.reshape(n_items[0], n_rows * n_cols)
            else:
                res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8)
                res = res.reshape(n_items[0], 1)
        return res


    # one-hot encode a 1-D array
    def one_hot_encode(array, num_of_classes):
        return np.eye(num_of_classes)[array.reshape(-1)]

    X_test = load_data('/dbfs/mnt/' + account_name + '/' + container_name + '/test-images.gz', False) / 255.0
    y_test = load_data('/dbfs/mnt/' + account_name + '/' + container_name + '/test-labels.gz', True).reshape(-1)
    ```

1. Let's quickly test and make sure our web service is working. Run the following code to see if it works. 
    ```python
    import requests
    import json

    # send a random row from the test set to score
    random_index = np.random.randint(0, len(X_test)-1)
    input_data = "{\"data\": [" + str(list(X_test[random_index])) + "]}"

    headers = {'Content-Type':'application/json'}

    resp = requests.post(service.scoring_uri, input_data, headers=headers)

    print("POST to url", service.scoring_uri)
    #print("input data:", input_data)
    print("label:", y_test[random_index])
    print("prediction:", resp.text)
    ```

1. You have now successfully deployed a scikit learn model using the Azure Machine Learning service! You will likely want to clean up the azure machine learning workspace in order to avoid charges. Navigate to the Azure Portal and find your workspace. The click on "Deployments" to delete the container we just deployed.  
