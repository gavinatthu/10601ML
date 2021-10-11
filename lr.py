import numpy as np
import sys
import time
from matplotlib import pyplot as plt

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

class LR():
    def __init__(self, train_in, valid_in, dict_in, metrics, epochs):
        self.train_in = train_in
        self.valid_in = valid_in
        self.dict_in = dict_in
        self.metrics = metrics
        self.epochs = epochs
        self.lr = 0.01                                 # 初始化学习率
        self.labels, self.feats = self.data_reading(train_in)
        self.validlabels, self.validfeats = self.data_reading(valid_in)

        print("Nums of samples:{}, Features:{}".format(len(self.labels), self.feats.shape[1]))
        

    def data_reading(self, input = None):
        '''
        data reading from input filename
        '''
        labels, feats = [], []

        with open(input,'r') as inf:
            lines = inf.readlines()
            for line in lines:
                line.replace('\n', '\t')
                labels.append(float(line.split("\t")[0]))
                feat = np.array(line.split("\t")[1:], dtype=float)  # dtype=int for model1?
                feats.append(feat)
        
        return np.array(labels), np.array(feats)

    def SGD(self, label, feat, w, lr):

        N = len(self.labels)
        pred = sigmoid(np.dot(feat, w.T))
        label = label.astype(np.float64)                            # float精度转换，否则下一步会出错
        w_new = w + lr/N * (label - pred) * feat
        pred = 1 if pred >= 0.5 else 0                          # 二值化pred
        return w_new, pred
    

    def train(self):
        '''
        training for n epochs with train_data
        '''
        
        bias = np.ones((len(self.labels),1),dtype=float)
        self.feats = np.concatenate((bias, self.feats), axis=1) # 添加bias到第0列, feats-> (N,M+1)=(350, 301)

        vbias = np.ones((len(self.validlabels),1),dtype=float)
        self.validfeats = np.concatenate((vbias, self.validfeats), axis=1)

        w = np.zeros(self.feats.shape[1])                  # 0初始化权重矩阵(M+1)维 (301, )
        self.lls = []
        self.vlls = []
        for epoch in range(self.epochs):
            ll = 0
            vll = 0

            for i in range(len(self.labels)):
                label, feat = self.labels[i], self.feats[i]
                ll += label*np.log(sigmoid(np.dot(feat,w.T)))+\
                    (1-label)*np.log(sigmoid(1-np.dot(feat,w.T)))
                w, _ = self.SGD(label, feat, w, self.lr)
            
            for i in range(len(self.validlabels)):
                label, feat = self.validlabels[i], self.validfeats[i]
                vll += label*np.log(sigmoid(np.dot(feat,w.T)))+\
                    (1-label)*np.log(sigmoid(1-np.dot(feat,w.T)))
               

            ll /= len(self.labels)
            vll /= len(self.validlabels)
            self.lls.append(-ll)  
            self.vlls.append(-vll)  

            #self.test()
        self.weight = w
        return None

    def test(self, test_in, test_out):
        '''
        test with test_data or valid_data
        '''
        labels, feats = self.data_reading(test_in)
        bias = np.ones((len(labels),1),dtype=float)
        feats = np.concatenate((bias, feats), axis=1) # 添加bias到第0列, feats-> (N,M+1)=(350, 301)

        preds = []
        for i in range(len(labels)):
            label, feat = labels[i], feats[i]
            _, pred = self.SGD(label, feat, self.weight, self.lr)
            preds.append(pred)

        cor = np.sum(preds == labels)
        print("Correct:{}/{}, err_rate:{}".format(cor, len(labels), 1-cor/len(labels),6))

        self.preds = preds
        self.format_out(test_out)

        return float(1-cor/len(labels))

    def format_out(self, test_out):

        with open(test_out, 'w') as f:
            for pred in self.preds:
                f.write(str(pred)+'\n')

        return None
    
    def llfigure(self):
        self.lls = np.array(self.lls)
        self.vlls = np.array(self.vlls)
        plt.figure()
        plt.plot(range(len(self.lls)),self.lls,label ='train')
        plt.plot(range(len(self.vlls)),self.vlls, label ='valid')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('negative log likelihood')
        plt.show()
        plt.savefig('ljw.png')
    
    


def main(train_in, valid_in, test_in, dict_in, train_out, test_out, metrics, epochs):
    start = time.time()
    myLR = LR(train_in, valid_in, dict_in, metrics, epochs)
    myLR.train()
    myLR.llfigure()
    acc_train = myLR.test(train_in, train_out)
    acc_test = myLR.test(test_in, test_out)

    with open(metrics, 'w') as f:
        f.write('error(train): ' + str("%.6f" % acc_train) + '\n')
        f.write('error(test): ' + str("%.6f" % acc_test) + '\n')

    print("Time consuming:", time.time() - start)
    #myLR.test(train_in, train_out)
    #myLR.test(test_in, test_out)



    return None

if __name__ == '__main__':
    '''
    python lr.py formatted_train.tsv formatted_valid.tsv formatted_test.tsv dict.txt train_out.labels test_out.labels metrics_out.txt 60
    '''
    '''
    ljw win用:
    python lr.py ./output/formatted_train.tsv ./output/formatted_valid.tsv `
         ./output/formatted_test.tsv ./handout/dict.txt `
         ./output/train_out.labels ./output/test_out.labels `
         ./output/metrics_out.txt 60
    gyx mac用:
    python3 lr.py ./output/formatted_train.tsv ./output/formatted_valid.tsv \
         ./output/formatted_test.tsv ./handout/dict.txt \
         ./output/train_out.labels ./output/test_out.labels \
         ./output/metrics_out.txt 50
    '''
    train_in = sys.argv[1]
    valid_in = sys.argv[2]
    test_in = sys.argv[3]
    dict_in = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics = sys.argv[7]
    epochs = int(sys.argv[8])

    main(train_in, valid_in, test_in, dict_in, train_out, test_out, metrics, epochs)