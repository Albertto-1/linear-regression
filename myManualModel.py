import numpy as np

class ManualLinearRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.ones(num_features).reshape(num_features,1)
        self.b = 1
    
    def forward_pass(self, X):
        y = self.b + np.dot(X, self.W)
        return y
    
    def compute_loss(self, y, y_true):
        loss = np.sum(np.square(y - y_true))
        return loss/(2*y.shape[0])
    
    def backward_pass(self, X, y_true, y_hat):
        m = y_hat.shape[0]
        db = np.sum(y_hat - y_true)/m
        dW = np.sum(np.dot(np.transpose(y_hat - y_true), X), axis=0)/m
        return dW, db
    
    def update_params(self, dW, db, lr):
        self.W = self.W - lr * np.reshape(dW, (self.num_features, 1))
        self.b = self.b - lr * db
    
    def train(self, x_train, y_train, iterations, lr):
        losses = []
        loss=100
        i=0
        while loss>0.2:
            y_hat = self.forward_pass(x_train)
            loss = self.compute_loss(y_hat, y_train)
            losses.append(loss)
            dW, db = self.backward_pass(x_train, y_train, y_hat)
            self.update_params(dW, db, lr)
            if i % 100 == 0:
                print('Iter: {}, Current loss: {:.4f}'.format(i, loss))
            i+=1
            if i==iterations:
                break
        return losses
