# LR
# house price prediction
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


'''----------load dataset-----------'''
dataset = datasets.load_boston()

# x features：['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS','RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
x = dataset.data

target = dataset.target
y = np.reshape(target,(len(target), 1))

#dataset 1:3 test：training
x_train, x_verify, y_train, y_verify = train_test_split(x, y, random_state=1)

'''
x_train的shape：(379, 13)
y_train的shape：(379, 1)
x_verify的shape：(127, 13)
y_verify 的shape：(127, 1)
'''

lr = linear_model.LinearRegression()
lr.fit(x_train,y_train)
lr_pred = lr.predict(x_verify)
# DT
regressor = DecisionTreeRegressor(random_state=0).fit(x_train,y_train)
dt_pred=regressor.predict(x_verify)



'''output'''
#30 samples
plt.xlim([0,30])
plt.plot( range(len(y_verify)), y_verify, 'r', label='y_verify')
plt.plot( range(len(lr_pred)), lr_pred, 'b', label='lr_pred')
plt.plot( range(len(dt_pred)), dt_pred, 'g--', label='dt_pred' )
plt.title('Linear Regression VS Decision Tree')
plt.legend()
plt.savefig('c:/work/lr-13.png')
plt.show()


'''MSE'''
print(lr.coef_)
print(lr.intercept_)
print("MSE:",metrics.mean_squared_error(y_verify,lr_pred))
print("MSE:",metrics.mean_squared_error(y_verify,dt_pred))


#Output R-Square
print(lr.score(x_train,y_train))
print(lr.score(x_verify,y_verify))



#In one figure
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


dataset = datasets.load_boston()

# x features：['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS','RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
x = dataset.data

target = dataset.target

y = np.reshape(target,(len(target), 1))

x_train, x_verify, y_train, y_verify = train_test_split(x, y, random_state=1)


lr = linear_model.LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_verify)
regressor = DecisionTreeRegressor(random_state=0).fit(x_train,y_train)
y_pred=regressor.predict(x_verify)



plt.xlim([0,50])
plt.plot( range(len(y_verify)), y_verify, 'r', label='y_verify')
plt.plot( range(len(y_pred)), y_pred, 'g--', label='y_predict' )
plt.title('sklearn: Linear Regression')
plt.legend()
plt.savefig('d:/lr-13.png')
plt.show()


print(lr.coef_)
print(lr.intercept_)
print("MSE:",metrics.mean_squared_error(y_verify,y_pred))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_verify,y_pred)))

print(lr.score(x_train,y_train))
print(lr.score(x_verify,y_verify))
