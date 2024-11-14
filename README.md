# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the California Housing dataset using fetch_california_housing().
2. Prepare the features 'x' (first 3 columns) and targets 'y' (column 6 and target).
3. Split the data into training and testing sets using train_test_split().
4. Initialize StandardScaler for scaling features and targets (scaler_x and scaler_y).
5. Scale the input features (x_train, x_test) using scaler_x.fit_transform() and scaler_x.transform().
6. Scale the target values (y_train, y_test) using scaler_y.fit_transform() and scaler_y.transform().
7. Initialize the SGDRegressor model with max_iter=1000 and tol=1e-3.
8. Wrap SGDRegressor with MultiOutputRegressor to handle multiple target variables.
9. Fit the model on the training data: multi_output_sgd.fit(x_train, y_train).
10. Predict the target values for the test set using the trained model: multi_output_sgd.predict(x_test).
11. Inverse transform the predicted values and actual values to their original scales.
12. Calculate the Mean Squared Error (MSE) between the predicted and actual target values.
13. Print the MSE and display the first five predicted values.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Vishnu K M 
RegisterNumber: 212223240185

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


data=fetch_california_housing()
x=data.data[:,:3]
y=np.column_stack((data.target,data.data[:, 6]))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.transform(x_test)
y_train=scaler_y.fit_transform(y_train)
y_test=scaler_y.transform(y_test)

sgd=SGDRegressor(max_iter =1000,tol= 1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train,y_train)
y_pred=multi_output_sgd.predict(x_test)
y_pred=scaler_y.inverse_transform(y_pred)
y_test=scaler_y.inverse_transform(y_test)
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)
print("\nPredictions:\n",y_pred[:5])

*/
```

## Output:
![image](https://github.com/user-attachments/assets/07c76fb3-084f-424d-8ff2-8cbeee7ef122)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
