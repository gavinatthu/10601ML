""" ReLU Layer """

import numpy as np

X = np.random.random((500, 10)) - 0.5
W = np.random.random((500, 10)) + 0.5
loss1 = - np.sum(X * W, axis=1) 
loss2 = np.log(np.sum(np.exp(X), axis=1))
loss = (loss1 + loss2)/X.shape[0]
print(loss)