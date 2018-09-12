# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:43:55 2018

@author: siddhant
"""
#Artificial Neural Network

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13 ].values
y = dataset.iloc[:, -1].values

#Encoding categorical variable (country,gender)..
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X1 = LabelEncoder()
X[:, 1] =  labelEncoder_X1.fit_transform(X[:, 1])

labelEncoder_X2 = LabelEncoder()
X[:, 2] =  labelEncoder_X2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding dummy variable trap
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling """Required in DeepLearning"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Making the ANN
#importing keras library
import keras
from keras.models import Sequential
from keras.layers import Dense

#Intializing the ANN
classifier = Sequential()
#Adding input layer and first hidden layer (output layer = (no_of_input+no_of_output)/2)
classifier.add(Dense(units = 6, init = 'uniform', activation = 'relu',input_dim = 11))
#Adding second hidden layer
classifier.add(Dense(units = 6, init = 'uniform', activation = 'relu'))
#Adding output layer(for more than 2 layers activation is softmax)
classifier.add(Dense(units = 1, init = 'uniform', activation = 'sigmoid'))
#Compiling the ANN(Applying stochaisting gradient descent)(for more than 2 outcomes loss is categorical_crossentropy)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#Fitting the ANN to train set
classifier.fit(X_train, y_train, batch_size = 10 , nb_epoch = 100)

#Prediction on Test
# Predicting the Test set results
#(y_pred will return probabilities, to comapre with y_test we will have to change it to 0 or 1)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""Single Preiction for customer with
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
customer = [[0.0,0,600,1,40,3,60000,2,1,1,50000]]
y_pred2 = classifier.predict(sc.transform(np.array(customer)))
y_pred2 = (y_pred2 > 0.5)


