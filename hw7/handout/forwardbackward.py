
import numpy as np
import sys

def data_reading(dir_in):

    with open(dir_in,'r') as inf:
        lines = inf.readlines()
        sequences = []
        sequence = []
        for line in lines:
            if line == '\n':
                sequences.append(sequence)
                sequence = []
            else:
                sequence.append(line.strip('\n').split('\t') )
    return sequences

def index_reading(idx_in):
    with open(idx_in,'r') as inf:
        lines = inf.readlines()
        item2idx = {x.strip('\n'):idx for idx, x in enumerate(lines)}
    return item2idx

class HMM():
    def __init__(self, validation_input, index_to_word,index_to_tag, hmminit, hmmemit,hmmtrans):
        self.sequences = data_reading(validation_input)
        self.w2i = index_reading(index_to_word)
        self.t2i = index_reading(index_to_tag)
        self.init = hmminit
        self.emit = hmmemit
        self.trans = hmmtrans

    def forward(self):
        alpha = np.zeros((len(self.sequences), len(self.t2i.keys())))
        for j in range(0, len(self.t2i.keys())):
            alpha[0][j] = self.init[j] * self.emit[j][self.w2i[self.sequences[0]]]
        if alpha.shape[0] > 1: #
            alpha[0] /= np.sum(alpha[0])
        for i in range(1, len(self.sequences)):
            for j in range(0, len(self.t2i.keys())):
                sumA = 0.0
                for k in range(0, len(self.t2i.keys())):
                    sumA += alpha[i - 1][k] * self.trans[k][j]
                alpha[i][j] = self.emit[j][self.w2i[self.sequences[i]]] * sumA
            if i != len(self.sequences) - 1:
                alpha[i] /= np.sum(alpha[i])
        alpha /= np.sum(alpha, axis = 1).reshape(alpha.shape[0], -1)


        return alpha
    
    def backward(self):
        beta = np.zeros((len(self.sequences), len(self.t2i.keys())))
        for j in range(0, len(self.t2i.keys())):
            beta[-1][j] = 1
        for i in range(len(self.sequences) - 2, -1, -1):
            for j in range(0, len(self.t2i.keys())):
                for k in range(0, len(self.t2i.keys())):
                    beta[i][j] += self.emit[k][self.w2i[self.sequences[i + 1]]] * beta[i + 1][k] * self.trans[j][k]
        beta /= np.sum(beta, axis = 1).reshape(beta.shape[0], -1)
       

        return beta
    
    def forwardbackward(self):
        generation = self.forward()*self.backward()
        pred_tag = generation.argmax(axis = 1)
        pred_tags = [i for i in pred_tag]
        print(pred_tag.shape)
        log_likelihood = np.log(np.sum(self.alpha[-1]))
        
        return pred_tags, log_likelihood


    def test(self):
        predictions = []
        total_likelihood = 0.0
        total_tags = 0
        correct = 0
        for sequence in self.sequences:
            #print(sequence)
            #words = [item.split(' ')[0] for item in sequence]
            words = [item[0] for item in sequence]

            #words = [x  for item in sequence for x in item]
            print(words)
            tags = [item[1] for item in sequence]
            pred_tags, log_likelihood = self.forwardbackward()
            predictions.append(["{}_{}".format(word, pred_tags) for word, pred_tags in zip(words, pred_tags)])
            total_likelihood += log_likelihood
            total_tags += len(tags)
            for tag, pred_tag in zip(tags, pred_tags):
                if tag == pred_tag:
                    correct += 1
        average = total_likelihood / float(len(self.sequences))
        accuracy = float(correct) / float(total_tags)
        
        return predictions, average, accuracy

def main(validation_input,index_to_word,index_to_tag, hmminit, hmmemit, hmmtrans, predicted_file, metric_file):
    init = np.loadtxt(hmminit)
    emit = np.loadtxt(hmmemit)
    trans = np.loadtxt(hmmtrans)
    hmm = HMM(validation_input, index_to_word,index_to_tag, init, emit,trans)
    predictions, average, accuracy = hmm.test()
    with open(predicted_file, "w") as f:
        lines = [" ".join(item) for item in predictions]
        f.write("\n".join(lines))
    with open(metric_file, "w") as f:
        f.write("Average Log-Likelihood: {}\n".format(average))
        f.write("Accuracy: {}".format(accuracy))


if __name__ == '__main__':
    '''
    mac用:
    python3 forwardbackward.py ./en_data/train.txt ./en_data/index_to_word.txt \
        ./en_data/index_to_tag.txt ./en_data/hmmint.txt \
        ./en_data/hmmemit.txt ./en_data/hmmtrans.txt \
        ./en_data/predicted.txt ./en_data/metrics.txt
    '''
    validation_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmminit = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted_file = sys.argv[7]
    metric_file = sys.argv[8]

    main(validation_input,index_to_word,index_to_tag, hmminit, hmmemit, hmmtrans, predicted_file, metric_file)
