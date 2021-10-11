import numpy as np

def softmax(x):
    x -= np.max(x,axis=1).reshape((x.shape[0],1))
    x = np.exp(x)
    tmp = np.sum(x, axis = 1)   
    x /= tmp.reshape((x.shape[0], 1))
    return x

def cost_function(t, h):
    E = -1.0 / 100 * np.sum(t * np.log(h)+ (1 - t) * np.log(1 - h))
    return E

def d_cross_function(x, t, h):
    return 1.0 / len(t) * (np.dot(x.T, (h - t)))


def  softmax_classifier(W, input, label, lamda):
    """
      Softmax Classifier

      Inputs have dimension D, there are C classes, a minibatch have N examples.
      (In this homework, D = 784, C = 10)

      Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - input: A numpy array of shape (N, D) containing a minibatch of data.
      - label: A numpy array of shape (N, C) containing labels, label[i] is a
        one-hot vector, label[i][j]=1 means i-th example belong to j-th class.
      - lamda: regularization strength, which is a hyerparameter.

      Returns:
      - loss: a single float number represents the average loss over the minibatch.
      - gradient: shape (D, C), represents the gradient with respect to weights W.
      - prediction: shape (N, 1), prediction[i]=c means i-th example belong to c-th class.
    """
    ############################################################################
    # TODO: Put your code here
    h = softmax(np.dot(input, W))
    loss = cost_function(label, h) + lamda/2 * np.linalg.norm(W)**2
    prediction = np.argmax(h, axis=1)
    gradient = d_cross_function(input, label, h) + lamda * W
    
    ############################################################################
    return loss, gradient, prediction
