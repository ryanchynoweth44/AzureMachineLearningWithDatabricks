import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import common as common
import pickle
from azureml.core import Run, Model



# NOTE: this allows us to do logging within our training script but still orchestrate it from another file. 
run = Run.get_context()


# load data and keep the transformations minimal
df = pd.read_csv("https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")
df = pd.get_dummies(df,prefix=['Pclass'], columns = ['Pclass'])
df = pd.get_dummies(df,prefix=['Sex'], columns = ['Sex'])
df = df.dropna()
df.head()

# identify a handful of training columns
label = 'Survived'
training_cols = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_male', 'Sex_female', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard']

# split datasets
x_train, x_test, y_train, y_test = train_test_split(df[training_cols].values, np.array(df[label], dtype=int), test_size=0.9)


# train model
mod = DecisionTreeClassifier(max_depth=5)
mod.fit(x_train, y_train)

# make sure we can predict a value
preds = mod.predict(x_test)


cm = confusion_matrix(y_test, preds)
common.plot_confusion_matrix(cm, ['0', '1'])
plt.savefig("outputs/confusion_matrix.png")


precision = precision_score(y_test,preds)
accuracy = accuracy_score(y_test,preds)

run.log('precision', precision)
run.log('accuracy', accuracy)



with open('outputs/decision_tree_model.pkl', 'wb') as f:
    pickle.dump(mod, f)


Model.register(run.experiment.workspace, 'outputs/decision_tree_model.pkl', 'decision_tree_model', tags={'precision': precision, 'accuracy': accuracy})