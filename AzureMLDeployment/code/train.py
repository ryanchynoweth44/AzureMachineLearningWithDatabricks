import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
import pickle


# load data and keep the transformations minimal
df = pd.read_csv("https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")
df = pd.get_dummies(df,prefix=['Pclass'], columns = ['Pclass'])
df = pd.get_dummies(df,prefix=['Sex'], columns = ['Sex'])
df = df.dropna()
df.head()


# identify a handful of training columns
label = 'Survived'
training_cols = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_male', 'Sex_female', 'Age', 'SibSp', 'Parch']

# separate datasets
x_train = df[training_cols].values
y_train = df[label].values

# train model
mod = DecisionTreeClassifier(max_depth=5)
mod.fit(x_train, y_train)

# make sure we can predict a value
mod.predict(x_train[0:1])

with open('outputs/decision_tree_model.pkl', 'wb') as f:
    pickle.dump(mod, f)

