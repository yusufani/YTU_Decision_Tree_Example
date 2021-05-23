#!/usr/bin/env python
# coding: utf-8

# # Trees and Forests
# 
# 
# 
# ## Training a Decision Tree Classifier

# ### Load Data From CSV File

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
from sklearn import preprocessing

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df = pd.read_csv('teleCust1000t.csv')
df.head(100)

# ### Data Visualization and Analysis
# 

# In[6]:


df['custcat'].value_counts()

# ### Feature set

# Lets define feature sets, X:

# In[7]:


df.columns

# To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:

# In[8]:

column_names = ['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender',
                'reside']
X = df[column_names].values  # .astype(float)
# X = df[['region', 'tenure', 'age']].values  # .astype(float)
X[0:5]

# In[9]:


y = df['custcat'].values
y[0:5]

# ### Train Test Split
# Out of Sample Accuracy is the percentage of correct predictions that the model makes on data that that the model has NOT been trained on. Doing a train and test on the same dataset will most likely have low out-of-sample accuracy, due to the likelihood of being over-fit.
# 
# It is important that our models have a high, out-of-sample accuracy, because the purpose of any model, of course, is to make correct predictions on unknown data. So how can we improve out-of-sample accuracy? One way is to use an evaluation approach called Train/Test Split. Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. After which, you train with the training set and test with the testing set.
# 
# This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset is not part of the dataset that have been used to train the data. It is more realistic for real world problems.

# In[10]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# In[ ]:


## Write your own code for decision tree by using Information Gain 
## Don't use ready code 

# Import math Library
import math
import matplotlib.pyplot as plt


# In[ ]:


class DecisionTree():
    def __init__(self):
        self.tree = None

    def __entropy__(self, data):
        values, counts = np.unique(data, return_counts=True)
        entropy = 0.0
        for count in counts:
            fraction = count / len(data)
            entropy += -1 * fraction * math.log2(fraction)

        return entropy

    def __info_gain_weighted__(self, data, left, right):

        info = 0.0
        info += len(left) / len(data) * self.__entropy__([i[1] for i in left])
        info += len(right) / len(data) * self.__entropy__([i[1] for i in right])
        return info

        #        "left_part": left_part, "right_part": right_part}

        # del max_gain["val"]  # Unnecessary

        """
        max_gain["left_part"] = self.__create_node(left_part)
        

        # Right part may not contains any element So we can give class directly
        max_gain["right_part"] = self.__create_node(right_part)
        """

    def select_feature(self, X_train, y_train):
        dataset_entropy = self.__entropy__(y_train)

        columns = []

        for column in X_train.T:
            # trees[column] = {}
            node = {}

            zipped = zip(column, y_train)
            zipped = list(zipped)
            res = sorted(zipped, key=lambda x: x[0])

            columns.append(self.get_gain_and_splitting_point(res, dataset_entropy))
            # self.__entropy__(res)

            # node = self.__create_node(res)
            # columns.append(node)

        max_gain = -1
        max_id = -1
        max_point = -1
        for idx, column in enumerate(columns):
            if column["gain"] > max_gain:
                max_gain = column["gain"]
                max_id = idx
                max_point = column["split_point"]

        return max_id, max_point

    def check_for_leaf(self, data):
        X_train, y_train = np.hsplit(data, [-1])
        y_train = y_train.reshape(-1, )
        y_vals = np.unique(y_train)

        if len(y_vals) <= 1:  # If there is no meaning for splitting
            return y_vals[0]
        # elif   data.shape[1] == 0: # If there is only one column for splitting

        else:
            return self.rec_fit(X_train, y_train)

    def fit(self, X_train, y_train):
        self.tree = self.rec_fit(X_train, y_train)

    def rec_fit(self, X_train, y_train):
        tree = {}
        # print(X_train.shape)
        # print(X_train , y_train)
        column_id, splitting_point = self.select_feature(X_train, y_train)
        # print(column_id , splitting_point)

        tree["condition"] = {"splitting_point": splitting_point,
                             "column_id": column_id}

        data_all = np.c_[X_train, y_train]

        left = data_all[data_all[:, column_id] <= splitting_point]
        # left = np.delete(left, id, axis=1)
        right = data_all[data_all[:, column_id] > splitting_point]
        # right = np.delete(right, id, axis=1)

        tree["left"] = self.check_for_leaf(left)
        tree["right"] = self.check_for_leaf(right)

        return tree

    def get_gain_and_splitting_point(self, data, dataset_entropy):
        _, indices = np.unique(data, return_index=True, axis=0)
        # print("Unique indices : ", indices, "data: ", data)

        """
        if len(indices) <= 1:  # If all class labels are same
            return dataset_entropy - 0  # BUrası yanlış olaiblir
        """
        max_gain = {"gain": -1, "split_point": None}

        for split_point in np.unique(data):
            # print("Split point " , split_point)
            left_part = [(key, category) for key, category in data if split_point >= key]
            # print("left_part :" , left_part)
            right_part = [(key, category) for key, category in data if split_point < key]
            # print("right_part:", right_part)

            # print("Weighted info gain : " , self.__info_gain_weighted__(data, left_part, right_part) )
            info_gain = dataset_entropy - self.__info_gain_weighted__(data, left_part, right_part)
            if max_gain["gain"] < info_gain:
                max_gain = {"gain": info_gain, "split_point": split_point}

        return max_gain

    def get_class(self, row):
        # print(row)
        tree = self.tree
        while True:
            # print(tree)
            col_id = tree["condition"]["column_id"]
            if row[col_id] <= tree["condition"]["splitting_point"]:
                if type(tree["left"]) != dict:
                    return tree["left"]
                else:
                    tree = tree["left"]
            else:

                if type(tree["right"]) != dict:
                    return tree["right"]
                else:
                    tree = tree["right"]

    def predict(self, X_test, y_test):
        if self.tree is None:
            print("Please train tree before predict")
        accuracy = 0
        for row, out in zip(X_test, y_test):

            predict_class = self.get_class(row)
            if predict_class == out:
                accuracy += 1

        print("accuracy", accuracy / len(X_test))

    def visualize_node(self, ax, node, depth, child_no, column_names, max_depth):
        if depth > max_depth:
            return
        if type(node) == dict:
            textstr = '\n'.join([
                "Threshold:" + str(node["condition"]["splitting_point"]),
                "Feature:\n" + column_names[node["condition"]["column_id"]],

            ]
            )
        else:
            textstr = "Class:" + str(node),

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=1)

        distances = 1.0 / 2 ** (depth-1)
        print("depth ", depth)
        print("distances : ", distances)
        print("child_no : ", child_no)

        x = distances + (distances * child_no ) - 0.05
        print("X: ", x)

        y = (1 - depth / 8 ) +  0.1
        print("Y: ", y)
        print(50 * "*")
        # place a text box in upper left in axes coords
        ax.text(x, y, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        self.visualize_node(ax, node["left"], depth + 1, child_no, column_names, max_depth)
        self.visualize_node(ax, node["right"], depth + 1, child_no + 1, column_names, max_depth)

    def visualize_tree(self, column_names, max_depth=4):

        plt.figure(figsize=(50, 50))
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        ax.plot([0, 100], [0, 100])
        self.visualize_node(ax, self.tree, 1, 0, column_names, max_depth)

        plt.show()


# %%
dec_tree = DecisionTree()

dec_tree.fit(X_train, y_train)

# %%
dec_tree.predict(X_test, y_test)
# %%
dec_tree.visualize_tree(column_names)

# %%
"""

'''
# Define the calculate entropy function
def calculate_entropy(df_label):
    classes, class_counts = np.unique(df_label, return_counts=True)
    entropy_value = np.sum([(-class_counts[i] / np.sum(class_counts)) * np.log2(class_counts[i] / np.sum(class_counts))
                            for i in range(len(classes))])
    return entropy_value


# Define the calculate information gain function
def calculate_information_gain(dataset, feature, label):
    # Calculate the dataset entropy
    dataset_entropy = calculate_entropy(dataset[label])
    values, feat_counts = np.unique(dataset[feature], return_counts=True)

    # Calculate the weighted feature entropy                                # Call the calculate_entropy function
    weighted_feature_entropy = np.sum(
        [(feat_counts[i] / np.sum(feat_counts)) * calculate_entropy(dataset.where(dataset[feature]
                                                                                  == values[i]).dropna()[label]) for i
         in range(len(values))])
    feature_info_gain = dataset_entropy - weighted_feature_entropy
    return feature_info_gain


def create_decision_tree( dataset, df, features, label, parent):
    df_datum = np.unique(df[label], return_counts=True)
    dataset_unique_data = np.unique(dataset[label])

    if len(dataset_unique_data) >= 1:
        return dataset_unique_data[0]
    elif len(dataset) == 0:
        return dataset_unique_data[np.argmax(df_datum[1])]
    elif len(features) == 0:
        return parent
    else:
        parent = dataset_unique_data[np.argmax(df_datum[1])]

        item_values = [calculate_information_gain(dataset, feature, label) for feature in features]

        optimum_feature_index = np.argmax(item_values)
        optimum_feature = features[optimum_feature_index]
        decision_tree = {optimum_feature: {}}
        features = [i for i in features if i != optimum_feature]
        for value in np.unique(dataset[optimum_feature]):
            min_data = dataset.where(dataset[optimum_feature] == value).dropna()
            min_tree = create_decision_tree(min_data, df, features, label, parent)
            decision_tree[optimum_feature][value] = min_tree

        return (decision_tree)


# Set the features and label
features = df.columns[:-1]
label = 'custcat'
parent = None

# Train the decision tree model
decision_tree = create_decision_tree(df, df, features, label, parent)

# Predict using the trained model
sample_data = {'glucose': 86, 'bloodpressure': 104}
test_data = pd.Series(sample_data)
'''
'''
prediction = predict_diabetes(test_data,decision_tree)
prediction
'''

# In[ ]:
dec_tree = DecisionTree()

trees = dec_tree.fit(X_train, y_train)

# In[ ]:
"""
"""
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix


decisiontree = DecisionTreeClassifier(random_state=0)
model = decisiontree.fit(X_train, y_train)

target_predicted=model.predict(X_test)
print("Accuracy", model.score(X_test, y_test))

matrix = confusion_matrix(y_test, target_predicted)
print("Class Confusion Matrix\n", matrix)

# create decision tree classifier using entropy
decisiontree_entropy = DecisionTreeClassifier(criterion='entropy', random_state=0)

model_entropy = decisiontree_entropy.fit(X_train, y_train)

target_predicted=model_entropy.predict(X_test)
print("Accuracy", model_entropy.score(X_test, y_test))

matrix = confusion_matrix(y_test, target_predicted)
print("Class Confusion Matrix\n", matrix)
from sklearn import tree
fig = plt.figure(figsize=(25,20))

tree.plot_tree(decisiontree_entropy,

                   filled=True)

fig.show()
"""
"""
# In[ ]:


#
# 
# ##  Visualizing a Decision Tree Model

# In[ ]:


import pydotplus
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
from sklearn import tree

## Write your own code to visualize tree with 4 levels
"""
