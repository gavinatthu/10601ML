import numpy as np
import sys


def data_reading(dir_in):
    '''
    data reading from input filename
    '''
    labels, feats = [], []

    with open(dir_in,'r') as inf:
        lines = inf.readlines()
        for line in lines:
            a = np.eye(4)
            # label 转换为ono-hot形式
            idx = np.array(line.split(",")[0],dtype=float)
            labels.append(a[int(idx)])
            feats.append(line.split(",")[1:])
    return np.array(labels), np.array(feats,dtype=float)

class SGD():
    def __init__(self, lr):
        self.lr = lr

    def step(self, model):
        layers = model.layerList
        for layer in layers:
            if type(layer) == LinearLayer: #只有线形层会更新权重
                if layer.s_W is None:
                    # Adagrad 初始化
                    layer.s_W = np.zeros_like(layer.grad_W)
                    layer.s_b = np.zeros_like(layer.grad_b)

                layer.s_W += layer.grad_W * layer.grad_W
                layer.s_b += layer.grad_b * layer.grad_b
                
                diff_W = -self.lr / np.sqrt(layer.s_W + 1e-5) * layer.grad_W
                diff_b = -self.lr / np.sqrt(layer.s_b + 1e-5) * layer.grad_b

                layer.weight += diff_W
                layer.bias += diff_b


class SigmoidLayer():
    def forward(self, a):
        self.z = 1. / (1. + np.exp(-a))
        return self.z

    def backward(self, grad):
        return self.z * (1. - self.z) * grad


class SoftmaxCELayer():

    def forward(self, b, label):
        """
        b: (1, K)
        label:(1, K)
        """
        y_hat = np.exp(b) / np.sum(np.exp(b), axis=1)

        log_y = np.log(y_hat)
        self.loss = - np.sum(label * log_y)          # 按元素乘

        # CE梯度证明：https://blog.csdn.net/jasonleesjtu/article/details/89426465
        self.grad = y_hat - label
        self.acc = np.argmax(y_hat) == np.argmax(label)

        return self.loss

    def backward(self):
        """
        grad: (1, K)
        """
        return self.grad


class LinearLayer():
    def __init__(self, dim_input, dim_output, ini_flag):

        self.ini_flag = ini_flag
        if ini_flag == 1:
            self.weight = np.random.rand(dim_input, dim_output)
            self.bias = np.random.rand(1, dim_output)

            # self.weight = np.random.rand(dim_input+1, dim_output)

        elif ini_flag == 2:
            self.weight = np.zeros((dim_input, dim_output))
            self.bias = np.zeros((1, dim_output))
        else:
            print("invalid ini_flag")
            self.weight = None
        self.s_W, self.s_b = None, None

    def forward(self, input):
        """
        input: (1, dim_input)
        forward: (1, dim_output) 
        """
        self.input = input

        return np.dot(input, self.weight) + self.bias       # (1, dim_output)

    def backward(self, grad):
        """
        grad: (1, dim_output)
        backward: (1, dim_input)
        """

        self.grad_W = np.dot(self.input.T, grad)            # (dim_input, dim_output)
        self.grad_b = grad                                  # (1, dim_output)

        return np.dot(grad, self.weight.T)                  # (1, dim_input)


class TotalNet():
    def __init__(self, dim_input, hidden_units, ini_flag):
        num_class = 4
        self.FC1 = LinearLayer(dim_input, hidden_units, ini_flag)
        self.FC2 = LinearLayer(hidden_units, num_class, ini_flag)
        self.ACTFun = SigmoidLayer()

        self.layerList = []
        self.layerList.append(self.FC1)
        self.layerList.append(self.ACTFun)
        self.layerList.append(self.FC2)


    def forward(self, input):
        """
        从前向后传播
        """
        #output = self.FC2.forward(self.ACTFun.forward(self.FC1.forward(input)))

        for layer in self.layerList:
            input = layer.forward(input)
        return input
    
    def backward(self, grad):
        """
        从后向前反向传播
        """
        #grad = layer[0].backward(layer[1].backward(layer[2].backward(grad)))

        for layer in reversed(self.layerList):
            grad = layer.backward(grad)
        return grad
    

class NN():
    def __init__(self, train_in, epochs, hidden_units, ini_flag, lr):
        self.epochs = epochs

        self.labels, self.feats = data_reading(train_in)
        self.model = TotalNet(self.feats.shape[1], hidden_units, ini_flag)
        self.optimizer = SGD(lr)
        self.criterion = SoftmaxCELayer()


    def train(self):
        
        for epoch in range(self.epochs):
            loss_ls = []
            count = 0
            for i in range(len(self.labels)):
                # 每个sample一次迭代SGD
                input = np.expand_dims(self.feats[i], axis=0)
                label = np.expand_dims(self.labels[i], axis=0)

                # forward
                pred = self.model.forward(input)
                
                loss = self.criterion.forward(pred, label)

                # backward
                grad = self.criterion.backward()
                self.model.backward(grad)

                # update
                self.optimizer.step(self.model)

                # Record loss and accuracy
                loss_ls.append(loss)
                count += self.criterion.acc

            print("Epoch [{}][{}]\t Training Loss {:.4f}\t Accuracy {:.4f}".format(
                epoch, self.epochs, np.mean(loss), count/len(self.labels)))

        return loss_ls, count

    def test(self, dir_in):
        loss_ls, count = [], 0
        test_labels, test_feats = data_reading(dir_in)
        for i in range(len(test_labels)):

            # forward
            input = np.expand_dims(test_feats[i], axis=0)
            label = np.expand_dims(test_labels[i], axis=0)
            pred = self.model.forward(input)
            
            loss = self.criterion.forward(pred, label)
            # Record loss and accuracy
            loss_ls.append(loss)
            count += self.criterion.acc

        print("Testing Loss {:.4f}\t Accuracy {:.4f}".format(
            np.mean(loss), count/len(test_labels))) 
        return loss_ls, count




def main(train_in,val_in,train_out,val_out,metrics_out,epochs,hidden_units,ini_flag,lr):
    mynetwork = NN(train_in, epochs, hidden_units, ini_flag, lr)
    mynetwork.train()
    mynetwork.test(val_in)

    return None

if __name__ == '__main__':
    '''
    ljw win用:
    python neuralnet.py ./data/small_train.csv ./data/small_val.csv `
        ./output/small_train_out.labels ./output/small_val_out.labels `
        ./output/small_metrics.txt 2 4 2 0.1
    gyx mac用:
    python3 neoralnet.py ./data/small_train.csv ./data/small_val.csv \
        ./output/small_train_out.labels ./output/small_val_out.labels \
        ./output/small_metrics.txt 2 4 2 0.1
    '''
    train_in = sys.argv[1]
    val_in = sys.argv[2]
    train_out = sys.argv[3]
    val_out = sys.argv[4]
    metrics_out = sys.argv[5]
    epochs = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    ini_flag = int(sys.argv[8])
    lr = float(sys.argv[9])

    main(train_in,val_in,train_out,val_out,metrics_out,epochs,hidden_units,ini_flag,lr)