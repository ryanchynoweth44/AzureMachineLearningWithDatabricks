import json
import numpy as np
import os
import pickle
import logging


def init():
    global mod 
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'decision_tree_model.pkl')
    logging.log(20, "Model Path: {}".format(model_path))
    with open(model_path, 'rb') as f:
        mod = pickle.load(f)
    logging.log(20, "Model Loaded. ")


def run(data):
    data = np.array(json.loads(data)['data'])
    logging.log(20, data)
    # make prediction
    y_hat = mod.predict(data)
    logging.log(20, y_hat)
    # you can return any data type as long as it is JSON-serializable
    return y_hat.tolist()