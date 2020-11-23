from extract import getDataXandY
from sklearn.model_selection import train_test_split
from myManualModel import ManualLinearRegression
import numpy as np
import math
import matplotlib.pyplot as plt


X,y = getDataXandY()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=45)

# Setup the data
X_train = X_train.values
y_train = np.reshape(y_train.values, (y_train.shape[0], 1))
X_test = X_test.values
y_test = np.reshape(y_test.values, (y_test.shape[0], 1))

# Init the model
lin_reg = ManualLinearRegression(X.shape[1])
# Train the model
losses = lin_reg.train(X_train, y_train, 10000, 5e-5)

# plt.plot(losses)
# plt.show()
# Predict
y_pred = lin_reg.predict(X_test)
# Mean Square Error
mse = lin_reg.compute_loss(y_pred, y_test)
print('MSE: ', mse)

# # print(y_test)
# # print(y_pred)
# print((lin_reg.predict([[22,29.5,0,0,1,1,0,0,0,1,0]])[0]))
index=0
print('Pred:\t|\tReal:')
for i in range(index,index+10):
    print("{:.2f}".format(y_pred[i][0]), '\t|\t', "{:.2f}".format(y_test[i][0]))
