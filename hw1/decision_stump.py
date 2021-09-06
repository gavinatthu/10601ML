import numpy as np
from collections import Counter
import argparse
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


class Node():
    def __init__(self, input, output, depth=0, split_idx=0, father = None):
        self.left   = None
        self.right  = None
        self.father = father
        self.llable = None
        self.rlable = None
        self.input = input
        self.output = output
        self.label = Counter(output).most_common(1)[0][0]
        self.depth = depth
        self.split_idx = split_idx


    def splitNode(self):
        '''
        分割特征到下一层节点
        '''
        idx = self.split_idx
        feats = self.input[:, idx]
        feat1, feat2 = np.unique(feats)

        if self.output[feats == feat1].any():
            self.left = Node(self.input[feats == feat1, :], self.output[feats == feat1], \
                        self.depth + 1, self.split_idx, self)
            self.llable = feat1

        if self.output[feats == feat2].any():
            self.right= Node(self.input[feats == feat2, :], self.output[feats == feat2], \
                        self.depth + 1, self.split_idx, self)
            self.rlable = feat2


    def train(self):
        '''
        训练决策树
        '''
        if (len(np.unique(self.output)) == 1) or (self.depth == 1):
            return 
        self.splitNode()
        if self.left: self.left.train() 
        if self.right: self.right.train() 
        return
    

    def predictRow(self, X_row):
        '''
        递归当前节点样本
        '''
        if (not self.left) and (not self.right):
            return self.label
        elif X_row[self.split_idx] == self.llable:
            return self.left.predictRow(X_row)
        elif X_row[self.split_idx] == self.rlable:
            return self.right.predictRow(X_row)
        elif (not self.left) or (not self.right):
            return self.label

    
    def predict(self, testX):
        '''
        通过测试集预测类别
        '''
        outcome = []
        for row in testX:
            outcome.append(self.predictRow(row))
        return np.array(outcome)



def main(train_in, test_in, split_idx, train_out, test_out, metrics):
    # 验证路径
    print(train_in, test_in, split_idx, train_out, test_out, metrics)

    # 数据读取
    data, head = data_reading(train_in)
    label = data[:,-1]
    feat = data[:,:-1]
    test_data, _ = data_reading(test_in,)
    test_label = test_data[:,-1]

    # decisionstump 是depth=1 的decisiontree， 初始化

    root = Node(feat, label, 0, split_idx)

    root.train()
    
    # 通过训练好的decision tree 进行推断
    trainOutput = root.predict(data)
    testOutput  = root.predict(test_data)
    
    # 计算预测准确率
    trainErr = np.sum(trainOutput != label) / len(trainOutput)
    testErr = np.sum(testOutput != test_label) / len(testOutput)
    

    with open(train_out, 'w') as f:
        for i in trainOutput:   
            f.write(str(i) + '\n')
    with open(test_out, 'w') as f:
        for i in testOutput:   
            f.write(str(i) + '\n')
    with open(metrics, 'w') as f:
        f.write('error(train): ' + str(round(trainErr,6)) + '\n')
        f.write('error(test): ' + str(round(testErr,6)) + '\n')
    return None




if __name__ == '__main__':
    '''
    用argparse方法方便调试，用默认命令行参数代替每次输入，调试完成后删去本内容，改用之后sys.argv方法
    运行方法：
    python decisionStump.py
    
    args = argparse.ArgumentParser(description='Decision Stump')
    args.add_argument('--train_in', default='./data/education_train.tsv', type=str,
                    help='path to the training input .tsv file')
    args.add_argument('--test_in', default='./data/education_test.tsv', type=str,
                    help='path to the test input .tsv file')
    args.add_argument('--split_idx', default=5, type=int,
                    help='the index of feature at which we split the dataset')
    args.add_argument('--train_out', default='./output/education_5_train.labels', type=str,
                    help='path of output .labels file on training data')
    args.add_argument('--test_out', default='./output/education_5_test.labels', type=str,
                    help='path of output .labels file on test data')
    args.add_argument('--metrics', default='./output/education_5_metrics.txt', type=str,
                    help='path of the output .txt file to metrics')
    args = args.parse_args()
    main(args.train_in, args.test_in, args.split_idx, args.train_out, args.test_out, args.metrics)

    '''
    '''
    调试完成以后main部分改成以下内容
    运行方法：
    python decision_stump.py ./data/politicians_train.tsv ./data/politicians_test.tsv 0 ./output/politicians_0_train.labels ./output/politicians_0_test.labels ./output/politicians_0_metrics.txt
    '''

    train_in = sys.argv[1]
    test_in = sys.argv[2]
    split_idx = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics = sys.argv[6]
    main(train_in, test_in, split_idx, train_out, test_out, metrics)
    print("success!")
    
    

