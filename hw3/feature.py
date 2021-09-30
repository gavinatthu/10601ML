import numpy as np
import sys

def data_reading(filename):
    '''
    data reading from input filename
    '''
    data = []
    with open(filename,'r') as inf:
        for i in inf:
            data.append(i.strip('\n').split('\t'))
    data = np.array(data)
    label = data[:,0]
    sen = data[:,1]
    return label, sen


class Models():
    def __init__(self) -> None:
        pass

    def model1(self):
        return None

    def model2(self):
        return None


def main(train_in, valid_in, test_in, dict_in, train_out, valid_out, test_out, flag):
    train_label, train_data = data_reading(train_in)
    

    return None


if __name__ == '__main__':
    '''
    ljw win用:
    python feature.py ./handout/smalldata/train_data.tsv ./handout/smalldata/valid_data.tsv `
         ./handout/smalldata/test_data.tsv ./handout/dict.txtdict.txt `
         ./output/formatted_train.tsv ./output/formatted_valid.tsv `
         ./output/formatted_test.tsv 1

    gyx mac用:
    python3 feature.py ./handout/smalldata/train_data.tsv ./handout/smalldata/valid_data.tsv \
         ./handout/smalldata/test_data.tsv ./handout/dict.txtdict.txt \
         ./output/formatted_train.tsv ./output/formatted_valid.tsv \
         ./output/formatted_test.tsv 1
    '''

    train_in = sys.argv[1]
    valid_in = sys.argv[2]
    test_in = sys.argv[3]
    dict_in = sys.argv[4]
    train_out = sys.argv[5]
    valid_out = sys.argv[6]
    test_out = sys.argv[7]
    flag = sys.argv[8]          # 1 or 2: whether to construct the Model 1 or  Model 2

    main(train_in, valid_in, test_in, dict_in, train_out, valid_out, test_out, flag)