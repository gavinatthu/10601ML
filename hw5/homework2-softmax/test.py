import os,sys
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import mnist_data_loader


mnist_dataset = mnist_data_loader.read_data_sets("./MNIST_data/", one_hot=True)


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

'''
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
    #loss = cost_function(label, h) + lamda/2 * np.linalg.norm(W)**2
    loss = cost_function(label, h)
    prediction = np.argmax(h, axis=1)
    #gradient = d_cross_function(input, label, h) + lamda * W
    gradient = d_cross_function(input, label, h)
    ############################################################################
    return loss, gradient, prediction
'''

# training dataset
train_set = mnist_dataset.train 
# test dataset
test_set = mnist_dataset.test   

train_size = train_set.num_examples
test_size = test_set.num_examples
example_id = 0 

image = train_set.images[example_id] # shape = 784 (28*28)
label = train_set.labels[example_id] # shape = 10
#print(image)
plt.imshow(np.reshape(image,[28,28]))
batch_size = 100
max_epoch = 10
learning_rate = 0.001

# For regularization
lamda = 0.5
W = np.random.randn(28*28, 10) * 0.001
#W = np.zeros((784, 10))

loss_set = []
accu_set = []
disp_freq = 100

# Training process
for epoch in range(0, max_epoch):
    iter_per_batch = train_size // batch_size
    for batch_id in range(0, iter_per_batch):
        batch = train_set.next_batch(batch_size) # get data of next batch
        input, label = batch

        # softmax_classifier
        h = softmax(np.dot(input,W))
        #loss = cost_function(label, h) + lamda/2 * np.linalg.norm(W)**2
        loss = cost_function(label, h)
        prediction = np.argmax(h, axis=1)
        #gradient = d_cross_function(input, label, h) + lamda * W
        gradient = d_cross_function(input, label, h)


        label = np.argmax(label, axis=1) # scalar representation
        #print(prediction, label)
        #accuracy = sum(1.0 * (prediction - label)) / float(len(label))
        accuracy = sum(prediction == label) / float(len(label))
        loss_set.append(loss)
        accu_set.append(accuracy)
        
        # Update weights
        W = W - (learning_rate * gradient)

        if batch_id % disp_freq == 0:
            print("Epoch [{}][{}]\t Batch [{}][{}]\t Training Loss {:.4f}\t Accuracy {:.4f}".format(
                epoch, max_epoch, batch_id, iter_per_batch, 
                loss, accuracy))
    print()

correct = 0
iter_per_batch = test_size // batch_size

# Test process
for batch_id in range(0, iter_per_batch):
    batch = test_set.next_batch(batch_size)
    data, label = batch
    
    # We only need prediction results in testing
    _,_, prediction = softmax_classifier(W, data , label, lamda)
    label = np.argmax(label, axis=1)
    correct += sum(prediction == label)
    
accuracy = correct * 1.0 / test_size
print('Test Accuracy: ', accuracy)
# training loss curve
plt.figure()
plt.plot(loss_set, 'b--')
plt.xlabel('iteration')
plt.ylabel('loss')
# training accuracy curve
plt.figure()
plt.plot(accu_set, 'r--')
plt.xlabel('iteration')
plt.ylabel('accuracy');