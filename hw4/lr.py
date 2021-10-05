import numpy as np
import sys


class LR():
    def __init__(self, train_in, dict_in, metrics, epochs):
        self.train_in = train_in
        self.dict_in = dict_in
        self.metrics = metrics
        self.epochs = epochs
        self.labels, self.feats = self.data_reading()
        print(self.labels.shape, self.feats.shape)
        

    def data_reading(self):
        '''
        data reading from input filename
        '''
        labels, feats = [], []

        with open(self.train_in,'r') as inf:
            lines = inf.readlines()
            for line in lines:
                line.replace('\n', '\t')
                labels.append(line.split("\t")[0])
                feat = np.array(line.split("\t")[1:], dtype=float)  # dtype=int for model1?
                feats.append(feat)
        
        return np.array(labels), np.array(feats)

    def gradient(self, input):
        grad = 0
        return grad

    def train(self):
        '''
        training for n epochs with train_data
        '''
        cor = 0
        for epoch in range(self.epochs):
            for label, 
            if pred == label:
                cor += 1 

        return None

    def test(self):
        '''
        test with test_data or valid_data
        '''
        return None



def main(train_in, valid_in, test_in, dict_in, train_out, test_out, metrics, epochs):
    myLR = LR(train_in, dict_in, metrics, epochs)
    myLR.train()

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
         ./output/metrics_out.txt 60
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