# Databricks notebook source
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

container_name = ""
account_name = ""


# set aml workspace parameters here. 
subscription_id = ""
resource_group = ""
workspace_name = ""
workspace_region = ""

ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)


# create experiment
experiment_name = 'demo_mnist'
exp = Experiment(workspace=ws, name=experiment_name)


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


# Load data
X_train = load_data('/dbfs/mnt/' + account_name + '/' + container_name + '/train-images.gz', False) / 255.0
y_train = load_data('/dbfs/mnt/' + account_name + '/' + container_name + '/train-labels.gz', True).reshape(-1)
X_test = load_data('/dbfs/mnt/' + account_name + '/' + container_name + '/test-images.gz', False) / 255.0
y_test = load_data('/dbfs/mnt/' + account_name + '/' + container_name + '/test-labels.gz', True).reshape(-1)


# function to generate an image
def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt
# get an image
img = gen_image(X_train[0])

# save image as png
img.savefig('/dbfs/mnt/' + account_name + '/' + container_name + '/sample_mnist_img.png', mode="overwrite")
plt.close()

# open png and display
from pyspark.ml.image import ImageSchema
image_df = ImageSchema.readImages('/mnt/' + account_name + '/' + container_name + '/sample_mnist_img.png')
display(image_df)


# start the run
run = exp.start_logging()

# train a model
clf = LogisticRegression()
clf.fit(X_train, y_train) 

# predict on test
y_hat = clf.predict(X_test)

# calculate accuracy on the prediction
acc = np.average(y_hat == y_test)
    
run.log('accuracy', np.float(acc))
os.makedirs('outputs', exist_ok=True)

# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=clf, filename='outputs/sklearn_mnist_model.pkl')


# upload the model file explicitly into artifacts 
run.upload_file(name = 'sklearn_mnist_model.pkl', path_or_stream = 'outputs/sklearn_mnist_model.pkl')
# register the model 
run.register_model(model_name = 'sklearn_mnist_model.pkl', model_path = 'outputs/sklearn_mnist_model.pkl' )

# save model to mounted directory as latest folder
dbutils.fs.cp("file:" + os.getcwd() + "/outputs/sklearn_mnist_model.pkl", '/dbfs/mnt/' + account_name + '/' + container_name + '/models/latest/sklearn_mnist_model.pkl', True)

run.take_snapshot('outputs')
# Complete the run
run.complete()
