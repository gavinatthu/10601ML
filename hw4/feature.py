import numpy as np
import sys

class Feat_Eng():
    def __init__(self, flag, dir_in, dir_out, dict_in, w2v_in):
        self.flag = flag
        self.dir_in = dir_in
        self.dir_out = dir_out
        self.dict_in = dict_in
        self.w2v_in = w2v_in

        #Data reading
        self.labels, self.sens = self.data_reading()
        self.dicts = self.dict_reading()


        #Choose feature model
        if flag == 1:
            self.feats = self.model1()
        elif flag == 2:
            self.w2c = self.w2c_reading()
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

        with open(self.dict_in,'r') as inf:
            lines = inf.readlines()
            for line in lines:
                key, value = line.split(" ")
                dicts[key] = value[:-1]                         #这里原本每一个value以\n结尾，舍弃

        return dicts

    def w2c_reading(self):
        '''
        w2c reading from input dict filename
        '''
        w2c = {}

        with open(self.w2v_in,'r') as inf:
            lines = inf.readlines()
            for line in lines:
                line.replace('\n', '\t')                        # 将句子尾部\n替换掉
                key = line.split("\t")[0]
                value = np.array(line.split("\t")[1:], dtype=float)
                w2c[key] = value

        return w2c

    def model1(self):
        print("Choose Model 1")

        feats = np.zeros((len(self.sens), len(self.dicts)), dtype=int)   #  (350, 14164)

        for i, sen in enumerate(self.sens):
            for j, key in enumerate(self.dicts.keys()):
                if key in sen:
                    feats[i][j] = 1                             # 用numpy原地修改耗时 31.0s 用list+append耗时 31.5 s

        print("shape of feats:", feats.shape)
        return feats



    def model2(self):
        print("Choose Model 2")

        feats = []      #  (350, 14164)
        #print(self.w2c.keys())

        for sen in self.sens:
            feat = []
            for word in sen:
                if word in self.w2c.keys():
                    feat.append(self.w2c[word])
            feat = np.array(feat)
            feats.append(np.mean(feat,axis=0))                  # 用numpy沿axis0求平均——P18公式

        feats = np.array(feats)
        print("shape of feats:", feats.shape)
        return feats
    
    def format_out(self):

        if self.flag == 1:
            with open(self.dir_out, 'w') as f:
                for i, feat in enumerate(self.feats):
                    string = '\t'.join(str(i) for i in feat.tolist())
                    f.write(str(self.labels[i])+'\t'+string+'\n')

        elif self.flag == 2:
            with open(self.dir_out, 'w') as f:
                for i, feat in enumerate(self.feats):
                    string = '\t'.join(format(i, '.6f') for i in feat.tolist())
                    f.write(format(float(self.labels[i]), '.6f')+'\t'+string+'\n')
        
        return None



def main(train_in, valid_in, test_in, dict_in, train_out, valid_out, test_out, flag, w2v_in):
    train_feat = Feat_Eng(flag, train_in, train_out, dict_in, w2v_in)
    valid_feat = Feat_Eng(flag, valid_in, valid_out, dict_in, w2v_in)
    test_feat = Feat_Eng(flag, test_in, test_out, dict_in, w2v_in)


    return None


if __name__ == '__main__':
    '''
    ljw win用:
    python feature.py ./handout/smalldata/train_data.tsv ./handout/smalldata/valid_data.tsv `
         ./handout/smalldata/test_data.tsv ./handout/dict.txt `
         ./output/formatted_train.tsv ./output/formatted_valid.tsv `
         ./output/formatted_test.tsv 2 ./handout/word2vec.txt

    gyx mac用:
    python3 feature.py ./handout/smalldata/train_data.tsv ./handout/smalldata/valid_data.tsv \
         ./handout/smalldata/test_data.tsv ./handout/dict.txt \
         ./output/formatted_train.tsv ./output/formatted_valid.tsv \
         ./output/formatted_test.tsv 2 ./handout/word2vec.txt
    '''

    train_in = sys.argv[1]
    valid_in = sys.argv[2]
    test_in = sys.argv[3]
    dict_in = sys.argv[4]
    train_out = sys.argv[5]
    valid_out = sys.argv[6]
    test_out = sys.argv[7]
    flag = int(sys.argv[8])          # 1 or 2: whether to construct the Model 1 or  Model 2
    w2v_in = sys.argv[9]
    main(train_in, valid_in, test_in, dict_in, train_out, valid_out, test_out, flag, w2v_in)