from extract import getDataXandY
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import math

X,y = getDataXandY()
# X,y = getDataXandY('carprice.csv')
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=45)

# Setup the data
X_train = X_train.values
y_train = np.reshape(y_train.values, (y_train.shape[0], 1))
X_test = X_test.values
y_test = np.reshape(y_test.values, (y_test.shape[0], 1))

# Init the model
lin_reg = LinearRegression()
# Train the model
lin_reg.fit(X_train,y_train)



# Predict
y_pred = lin_reg.predict(X_test)
# Mean Square Error
mse = mean_squared_error(y_test, y_pred)
print('MSE: ', mse)

# print(y_test)
# print(y_pred)
# print((lin_reg.predict([[22,29.5,0,0,1,1,0,0,0,1,0]])[0]))
index=0
print('Pred:\t|\tReal:')
for i in range(index,index+10):
    print("{:.2f}".format(y_pred[i][0]), '\t|\t', "{:.2f}".format(y_test[i][0]))
