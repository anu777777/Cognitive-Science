#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[2]:


data = load_iris()


# In[3]:


X=data.data
y=data.target


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=4)


# In[5]:


learning_rate = 0.1
iterations = 5000
N = y_train.size


# In[8]:


input_size = 4
hidden_size = 2
output_size = 3


# In[10]:


np.random.seed(10)

# Hidden layer
W1 = np.random.normal(scale=0.5, size=(input_size, hidden_size))

# Output layer
W2 = np.random.normal(scale=0.5, size=(hidden_size, output_size))


# In[12]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mean_squared_error(y_pred, y_true):
    # One-hot encode y_true (i.e., convert [0, 1, 2] into [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_true_one_hot = np.eye(output_size)[y_true]

    # Reshape y_true_one_hot to match y_pred shape
    y_true_reshaped = y_true_one_hot.reshape(y_pred.shape)

    # Compute the mean squared error between y_pred and y_true_reshaped
    error = ((y_pred - y_true_reshaped) ** 2).sum() / (2 * y_pred.size)

    return error


# In[13]:


def accuracy(y_pred, y_true):
    acc = np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)
    return acc.mean()

results = pd.DataFrame(columns=["mse", "accuracy"])


# In[15]:


results = pd.DataFrame(columns=["mse", "accuracy"])

for itr in range(iterations):
    # Feedforward propagation
    Z1 = np.dot(X_train, W1)
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)

    # Calculate error
    mse = mean_squared_error(A2, y_train)
    acc = accuracy(np.eye(output_size)[y_train], A2)

    new_row = pd.DataFrame({"mse": [mse], "accuracy": [acc]})
    results = pd.concat([results, new_row], ignore_index=True)

results.mse.plot(title="Mean Squared Error")
plt.show()

results.accuracy.plot(title="Accuracy")
plt.show()


# In[ ]:




