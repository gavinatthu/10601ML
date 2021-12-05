
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
        self.i2t = {i: t for t, i in self.t2i.items()}
        self.init = hmminit
        self.emit = hmmemit
        self.trans = hmmtrans

    def forward(self, words):
        alpha = np.empty((len(words), len(self.t2i.keys())))
        word = words[0]
        word_ind = self.w2i[word]
        alpha[0, :] = self.init * self.emit[:, word_ind]
        for i in range(1, len(words)):
            word = words[i]
            word_id = self.w2i[word]
            alpha[i, :] = self.emit[:, word_id] * np.dot(alpha[i - 1, :], self.trans)
            """
            last_ = i - 1
            for j in range(alpha.shape[1]):
                s = 0
                for k in range(alpha.shape[1]):
                    s += alpha[last_, k] * self.trans[k, j]
                alpha[i, j] = self.emit[j, word_id] * s
            """
        return alpha
    
    def backward(self, words):
        beta = np.empty((len(words), len(self.t2i.keys())))
        beta[len(words) - 1, :] = 1
        i = len(words) - 2
        while i >= 0:
            word = words[i + 1]
            word_id = self.w2i[word]
            beta[i, :] = np.dot(self.emit[:, word_id] * beta[i + 1, :], self.trans.T)
            """
            next_ = i + 1
            for j in range(beta.shape[1]):
                s = 0
                for k in range(beta.shape[1]):
                    s += self.emit[k, word_id] * beta[next_, k] * self.trans[j, k]
                beta[i, j] = s
            """
            i -= 1
        return beta
    
    def forwardbackward(self, words):
        if len(words) == 0:
            return [], 0
        alpha = self.forward(words)
        beta = self.backward(words)
        generation = alpha * beta
        pred_tag_i = generation.argmax(axis=1)
        pred_tag = [self.i2t[i] for i in pred_tag_i]
        s = alpha[-1].sum()
        log_likelihood = np.log(s) if s != 0 else 0
        return pred_tag, log_likelihood

    def test(self):
        predictions = []
        total_likelihood = 0.0
        total_tags = 0
        correct = 0
        for sequence in self.sequences:
            words, tags = zip(*sequence)
            pred_tags, log_likelihood = self.forwardbackward(words)
            predictions.append(["{}\t{}\n".format(word, pred_tags) for word, pred_tags in zip(words, pred_tags)])
            total_likelihood += log_likelihood
            total_tags += len(tags)
            for tag, pred_tag in zip(tags, pred_tags):
                if tag == pred_tag:
                    correct += 1
        average = (total_likelihood / float(len(self.sequences))) if self.sequences else 0
        accuracy = (float(correct) / float(total_tags)) if total_tags else 0
        
        return predictions, average, accuracy

def main(validation_input,index_to_word,index_to_tag, hmminit, hmmemit, hmmtrans, predicted_file, metric_file):
    init = np.loadtxt(hmminit)
    emit = np.loadtxt(hmmemit)
    trans = np.loadtxt(hmmtrans)
    hmm = HMM(validation_input, index_to_word,index_to_tag, init, emit,trans)
    predictions, average, accuracy = hmm.test()
    lines = []
    for prediction in predictions:
        lines.extend(prediction)
        lines.append("\n")
    with open(predicted_file, "w") as f:
        f.writelines(lines)
    with open(metric_file, "w") as f:
        f.write("Average Log-Likelihood: {}\n".format(average))
        f.write("Accuracy: {}".format(accuracy))


if __name__ == '__main__':
    '''
    mac用:
    python3 forwardbackward.py ./en_data/validation.txt ./en_data/index_to_word.txt \
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
