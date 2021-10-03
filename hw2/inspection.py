import numpy as np
from collections import Counter
import sys



def data_reading(filename):
    '''
    数据读取
    '''
    data = []
    inf = open(filename)
    for i in inf:
        data.append(i.strip('\n').split('\t'))
    output = np.array(data[1:]).astype(object)
    head = np.array(data[0])
    return output, head

def Entropy(labels):
    '''
    the entropy of the labels before any splits
    Calculated in bits using log based on 2
    '''

    nums = len(labels)
    counts = list(Counter(labels.tolist()).values())
    entropy = sum([- label / nums * np.log2(label / nums) for label in counts])
    return entropy


def Error(labels):
    counts = list(Counter(labels.tolist()).values())
    error = 1 - max(counts) / sum(counts)
    return error


def main(train_in, train_out):
    data, _ = data_reading(train_in)
    label = data[:,-1]
    feat = data[:,:-1]
    etp = Entropy(label)
    err = Error(label)
    print('entropy =', etp, '\nerror =', err)
    with open(train_out, 'w') as f:
        f.write('entropy: ' + str(etp) + '\n')
        f.write('error: ' + str(err) + '\n')

    return None

if __name__ == '__main__':

    '''
    运行方法：
    python3 inspection.py ./data/small_train.tsv ./output/small_inspect.txt
    '''

    train_in = sys.argv[1]
    train_out = sys.argv[2]

    main(train_in, train_out)
    print("success!")
