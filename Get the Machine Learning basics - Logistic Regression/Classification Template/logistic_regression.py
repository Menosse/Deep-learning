# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:30:48 2020

@author: Fernando

This Logistic regression model will consider only the age and estimated salary to predict if a user will buy or not a SUV

!! Note that Datapreprocessing consists in 6 Steps
1 - Import main libraries numpy, pandas and matplotlib.pyplot
2 - Import the dataset
3 - Take care of missing data. (It is possible to assume the mean value among the column for the missing data)
4 - Encode Categorical variables, like country, gender, etc
5 - Split the dataset into training and test sets
6 - Apply feature scailing for accurate prediction
"""
# ### ================================= ###
# ### 1 - Import the main libraries ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# ### ================================= ###
# ### 2 - Import the dataset ###

dataset = pd.read_csv("Social_Network_Ads.csv")
# Create Independent variable (IV) and dependent variable (DV)
# IV Age and estimated Salary
x = dataset.iloc[:, [2,3]].values
# DV Buy or not
y = dataset.iloc[:, 4].values



# ### ================================= ###
# ### 3 - Take care of missing data. ###

# # !! IN THIS MODEL IT IS NOT NECESSARY !!
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values='NaN', strategy='mean')
# # Define which columns will be changed by simple imputer
# imputer = imputer.fit(x[:, 1:3])
# x[:,1,3] = imputer.transform(x[:, 1:3])



# ### ================================= ###
# ### 4 - Encode Categorical variables

# # !! IN THIS MODEL IT IS NOT NECESSARY !!
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# # Encode boolean feature (yes/no, male/female) - This example encodes x and y
# labelencoder_x = LabelEncoder()
# x[:,0] = labelencoder_x.fit_transform(x[:,0])
# labelencoder_y = LabelEncoder()
# y = labelencoder_y.fit_transform(y)
# # Encode categorical variable with multiple values - categorical_features = which column to be encoded
# one_hot_encoder = OneHotEncoder(categories='auto', categorical_features = [0])
# x = one_hot_encoder.fit_transform(x).toarray()



# ### ================================= ###
# ### 5 - Create a training and test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 0)



# ### ================================= ###
# ### 6 - Feature scaling

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.fit_transform(x_test)



# ### ================================= ================================= ================================= ###
# ### =================================        LOGISTIC REGRESSION        ================================= ###
# ### ================================= ================================= ================================= ###

# Create the Logistic Regression classifier and fit to the trainset
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

# Predict the test set results
y_pred = classifier.predict(x_test)

# Evaluate the logistic regression classifier with CONFUSION MATRIX computation
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizing the training/test set Results
def PlotChart(x_set, y_set, setName):
    from matplotlib.colors import ListedColormap
    x1, x2 = np.meshgrid(np.arange(start = x_set[:,0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = x_set[:,1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red','green')))
    plt.xlim(x1.min(),x1.max())
    plt.ylim(x2.min(),x2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                    c = ListedColormap(('red','green'))(i), label = j)
    plt.title(f'Logistic Regression ({setName})')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

PlotChart(x_train, y_train, 'Training set')
PlotChart(x_test, y_test, 'Test set')
