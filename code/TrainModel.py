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

container_name = ""
account_name = ""

# set aml workspace parameters here. 
subscription_id = ""
resource_group = ""
workspace_name = ""
workspace_region = ""

# connect to workspace
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

# list of numbers from 0.0 to 1.0 with a 0.05 interval for regularization rates
regs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
save_date = str(datetime.datetime.now())

# vars to hold which regularization rate and accuracy is best
best_reg = 0
last_acc = -1 
acc_list = []
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
    
  # add the value to a dictionary to graph
  acc_list.append(acc)
  
  run.log('regularization rate', np.float(reg))
  run.log('accuracy', np.float(acc))
  os.makedirs('outputs', exist_ok=True)
  # note file saved in the outputs folder is automatically uploaded into experiment record
  joblib.dump(value=clf, filename='outputs/reg_' + str(reg) + '_sklearn_mnist_model.pkl')


# upload the model file explicitly into artifacts 
run.upload_file(name = 'sklearn_mnist_model.pkl', path_or_stream = 'outputs/reg_' + best_reg + '_sklearn_mnist_model.pkl')
# register the model 
run.register_model(model_name = 'sklearn_mnist_model.pkl', model_path = 'outputs/reg_' + best_reg + '_sklearn_mnist_model.pkl' )
run.tag("Regularization Rate", best_reg)

plt.plot(regs, acc_list)
plt.xlim([0, 1])
plt.ylim([0,1])
plt.savefig('/dbfs/mnt/' + account_name + '/' + container_name + '/acc_image.png', mode="overwrite")
plt.close()

run.upload_file(name = 'accuracies.png', path_or_stream = '/dbfs/mnt/' + account_name + '/' + container_name + '/acc_image.png')

run.take_snapshot('outputs')
# Complete the run
run.complete()
