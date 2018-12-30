# Train an MNIST Model in Azure Databricks

## Train Model
1. Import the required libraries 
    ```python 
    import os
    import datetime
    import azureml
    from azureml.core import Workspace, Run, Experiment
    from azureml.train.estimator import Estimator

    import gzip
    import numpy as np
    import struct

    from sklearn.linear_model import LogisticRegression
    from sklearn.externals import joblib


    import matplotlib
    import matplotlib.pyplot as plt

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

    # set container and account name for data sources
    container_name = ""
    account_name = ""
    ```

1. Next connect to your workspace. Please note that you may need to sign in interactively using a browser.  
    ```python
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
    ```

1. Create an experiment to track the runs in your workspace. A workspace can have muliple experiments.
    ```python 
    experiment_name = 'demomnist'

    exp = Experiment(workspace=ws, name=experiment_name)
    ```
1. Our data files are saved as gzip files so we will need a few helper functions to load and manipulate our data. Paste and run the following code to create python functions.  
    ```python 
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
    ```

1. Lets read our data from blob storage into python Arrays. 
    ```python
    # Load data
    X_train = load_data('/dbfs/mnt/user/blob/' + account_name + '/' + container_name + '/train-images.gz', False) / 255.0
    y_train = load_data('/dbfs/mnt/user/blob/' + account_name + '/' + container_name + '/train-labels.gz', True).reshape(-1)
    X_test = load_data('/dbfs/mnt/user/blob/' + account_name + '/' + container_name + '/test-images.gz', False) / 255.0
    y_test = load_data('/dbfs/mnt/user/blob/' + account_name + '/' + container_name + '/test-labels.gz', True).reshape(-1)
    ```

1. If you wish you can generate one of the images and display it in your notebook.  
    ```python
    # function to generate an image
    def gen_image(arr):
        two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
        plt.imshow(two_d, interpolation='nearest')
        return plt
    # get an image
    img = gen_image(X_train[0])

    # save image as png
    img.savefig('/dbfs/mnt/user/blob/' + account_name + '/' + container_name + '/sample_mnist_img.png', mode="overwrite")

    # open png and display
    from pyspark.ml.image import ImageSchema
    image_df = ImageSchema.readImages('/mnt/user/blob/' + account_name + '/' + container_name + '/sample_mnist_img.png')
    display(image_df)
    ```

1. Now lets train and test a machine learning model! We are going to train a simple logistic regression model for this example with changinge regularization rates. We save all models to a local output folder but only upload the model with the best accuracy. Please note that since we connected to our Azure Machine Learning Workspace that this model training is being tracked. I would recommend navigating the Azure Portal and click on your workspace.    
    ```python
    # list of numbers from 0.0 to 1.0 with a 0.05 interval for regularization rates
    regs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    save_date = str(datetime.datetime.now())

    best_reg = 0
    last_acc = -1 
    # start the run
    run = exp.start_logging()

    for reg in regs:
        print("starting reg: " + str(reg))
        # train a model
        clf = LogisticRegression(C=1.0/reg, random_state=42)
        clf.fit(X_train, y_train) 
        
        print("Predicting on test dataset")
        y_hat = clf.predict(X_test)
        # calculate accuracy on the prediction
        acc = np.average(y_hat == y_test)
        print('Accuracy is', acc)
        # track which model is best
        if acc > last_acc:
            best_reg = reg
        
        run.log('regularization rate', np.float(reg))
        run.log('accuracy', np.float(acc))
        os.makedirs('outputs', exist_ok=True)
        # note file saved in the outputs folder is automatically uploaded into experiment record
        joblib.dump(value=clf, filename='outputs/reg_' + reg + '_sklearn_mnist_model.pkl')

    # upload the best model file explicitly into artifacts 
    run.upload_file(name = 'sklearn_mnist_model.pkl', path_or_stream = 'outputs/reg_' + best_reg + '_sklearn_mnist_model.pkl')
    # register the model 
    run.register_model(model_name = 'sklearn_mnist_model.pkl', model_path = 'outputs/reg_' + best_reg + '_sklearn_mnist_model.pkl' )
    # tag the run with the best regularization rate
    run.tag("Regularization Rate", best_reg)

    run.take_snapshot('outputs')
    # Complete the run
    run.complete()
    ```

1. Because we are logging and saving everything to our workspace we can navigate the Azure Portal and look at our experiment runs. Here is an example of what mine looks like with two completed runs and one in progress. You can also navigate to the "Models" tab to see the model we just registered to the workspace as well.     
    ![](./imgs/02_AML_Workspace.png) 

1. We have now trained a machine learning model! We have tracked the training and saved our model to our Azure Machine Learning Workspace, therefore, we are ready to prepare and deploy our model! Please complete the [Deploy Model Walkthrough](03_DeployModel.md) to learn how to deploy a model in Azure Databricks and to Azure Kubernetes Service.  