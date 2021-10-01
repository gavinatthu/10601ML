import numpy as np
import sys


class Feat_Eng():
    def __init__(self, flag, dir_in, dir_out, dict_in):
        self.dir_in = dir_in
        self.dir_out = dir_out
        self.dict = dict_in

        #Data reading
        self.labels, self.sens = self.data_reading()
        self.dicts = self.dict_reading()
        print(len(self.dicts))

        #Choose feature model
        if flag == 1:
            self.feats = self.model1()
        elif flag == 2:
            self.feats = self.model2()
        else:
            print("invalid flad input")
        
        #Format Output
        self.format_out()


    def data_reading(self):
        '''
        data reading from input filename
        '''
        labels, sens = [], []

        with open(self.dir_in,'r') as inf:
            lines = inf.readlines()
            for line in lines:
                labels.append(line.split("\t")[0])              #按照"\t"切分得到每行为[label, sen]
                sens.append(line.split("\t")[1].split(" "))     #每一行句子sen按照空格" "切分成词汇

        return np.array(labels), np.array(sens, dtype="object")
    

    def dict_reading(self):
        '''
        dict reading from input dict filename
        '''
        dicts = {}

        with open(self.dict,'r') as inf:
            lines = inf.readlines()
            for line in lines:
                key, value = line.split(" ")
                dicts[key] = value[:-1]                         #这里原本每一个value以\n结尾，舍弃
        
        return dicts

    def model1(self):
        print("Choose Model 1")

        return None

    def model2(self):
        print("Choose Model 1")

        return None
    
    def format_out(self):

        return None



def main(train_in, valid_in, test_in, dict_in, train_out, valid_out, test_out, flag):
    train_feat = Feat_Eng(flag, train_in, train_out, dict_in)
    valid_feat = Feat_Eng(flag, valid_in, valid_out, dict_in)
    test_feat = Feat_Eng(flag, test_in, test_out, dict_in)


    return None


if __name__ == '__main__':
    '''
    ljw win用:
    python feature.py ./handout/smalldata/train_data.tsv ./handout/smalldata/valid_data.tsv `
         ./handout/smalldata/test_data.tsv ./handout/dict.txt `
         ./output/formatted_train.tsv ./output/formatted_valid.tsv `
         ./output/formatted_test.tsv 1

    gyx mac用:
    python3 feature.py ./handout/smalldata/train_data.tsv ./handout/smalldata/valid_data.tsv \
         ./handout/smalldata/test_data.tsv ./handout/dict.txt \
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
    flag = int(sys.argv[8])          # 1 or 2: whether to construct the Model 1 or  Model 2

    main(train_in, valid_in, test_in, dict_in, train_out, valid_out, test_out, flag)