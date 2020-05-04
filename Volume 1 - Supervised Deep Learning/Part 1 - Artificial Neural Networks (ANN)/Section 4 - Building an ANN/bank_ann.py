# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:08:29 2020

@author: Fernando

!! Note that Datapreprocessing consists in 6 Steps
1 - Import main libraries numpy, pandas and matplotlib.pyplot
2 - Import the dataset
3 - Take care of missing data. (It is possible to assume the mean value among the column for the missing data)
4 - Encode Categorical variables, like country, gender, etc
5 - Split the dataset into training and test sets
6 - Apply feature scailing for accurate prediction
"""
# ### ================================= ================================= ================================= ###
# ### =================================        Data Pre processing        ================================= ###
# ### ================================= ================================= ================================= ###
# ### ================================= ###
# ### 1 - Import the main libraries ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### ================================= ###
# ### 2 - Import the dataset ###
dataset = pd.read_csv("Churn_Modelling.csv")
# Create Independent variable (IV) and dependent variable (DV)
# IV Age and estimated Salary
x = dataset.iloc[:, 3:13].values
# DV Buy or not
y = dataset.iloc[:, 13].values



# ### ================================= ###
# ### 3 - Take care of missing data ###
# # !! IN THIS MODEL IT IS NOT NECESSARY !!
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values='NaN', strategy='mean')
# # Define which columns will be changed by simple imputer
# imputer = imputer.fit(x[:, 1:3])
# x[:,1,3] = imputer.transform(x[:, 1:3])



### ================================= ###
### 4 - Encode Categorical variables ###
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
# Encode boolean feature (yes/no, male/female) - This example encodes x and y
labelencoder_x1 = LabelEncoder()
x[:, 1] = labelencoder_x1.fit_transform(x[:, 1])
labelencoder_x2 = LabelEncoder()
x[:, 2] = labelencoder_x2.fit_transform(x[:, 2])
# Encode categorical variable with multiple values - categorical_features = which column to be encoded
onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = onehotencoder.fit_transform(x)
x = x[:, 1:]



# ### ================================= ###
# ### 5 - Split a training and test set ###
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)



# ### ================================= ###
# ### 6 - Feature scaling ###
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.fit_transform(x_test)



# ### ================================= ================================= ================================= ###
# ### =================================     Apply Deep Learning Model     ================================= ###
# ### ================================= ================================= ================================= ###
# ### ================================= ###
# ### 1 - Import Keras libraries and models ###
import keras
from keras.models import Sequential
from keras.layers import Dense


# ### ================================= ###
# ### 2 - Initialize the ANN ###
classifier = Sequential()



# ### ================================= ###
# ### 3 - Create input layer and hidden layers ###
# Create the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Create the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Add the output layer
#!!! If the dependent variable has more than 2 categories use ==> activation = 'softmax' <== !!!#
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# ### ================================= ###
# ### 4 - Compile the ANN###
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# ### ================================= ###
# ### 5 - Fit the classifier to the training set###
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)



# ### ================================= ###
# ### 6 - Make the predictions on the test set ###
y_pred = classifier.predict(x_test)
# Create the threshold and convert the predicted values to categorical data
y_pred = (y_pred > 0.5)



# ### ================================= ###
# ### 7 - Evaluate the logistic regression classifier with CONFUSION MATRIX computation ###
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



# ### ================================= ###
# ### HOME WORK - Predict a single new observation (customer) ###

new_pred = classifier.predict(scaler_x.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_pred = (new_pred > 0.5)
new_pred