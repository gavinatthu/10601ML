import numpy as np
from collections import Counter
import sys

def data_reading(filename):
    '''
    data reading from input dir
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
    

def loss(Y, X):
    '''
    决策树的负loss函数，希望选取loss最大的特征
    '''
    X_len = len(X)
    uni_featsm = np.unique(X)
    loss = Entropy(Y)
    for i in uni_featsm:
        loss -= (np.count_nonzero(X == i) / X_len) * Entropy(Y[X == i])
    return loss


class Node():
    def __init__(self, input, output, head, feat_idx, depth, max_depth, father = None):

        self.left   = None
        self.right  = None
        self.father = father
        self.llable = None
        self.rlable = None
        self.input = input
        self.output = output
        self.feat_idx = feat_idx
        # 首先按照出现次数 再按字典序反序排序 按照HW2-3.3要求
        self.label = sorted(Counter(output).items(), \
                        key = lambda pair: (pair[1], pair[0]))[-1][0]
        self.head = head
        self.depth = depth
        self.max_depth = max_depth


    def __str__(self):
        '''
        Overload print function
        '''
        ## 小顾来写吧:)
        string = str(Counter(self.output)[str(self.llable)])+' / '+ str(Counter(self.output)[str(self.rlable)])

        string_l = str(self.depth*" ")+ str(self.head[self.feat_idx])+' '+str(len(self.left.input))+\
            str(self.llable)
        string_r = str(self.depth*"  ")+str(len(self.right.input))+\
            str(self.rlable)
        return string + string_l+'\n'+string_r


    def Choose_Feat(self):
        """
        选择分类准确率最大的特征
        """
        mu_li = []
        for i in range(len(self.input[0])):
            mu_li.append(loss(self.output, self.input[:, i]))
        return mu_li.index(max(mu_li))


    def splitNode(self):
        '''
        分割特征到下一层节点
        '''
        
        self.feat_idx = self.Choose_Feat()
        feats = self.input[:, self.feat_idx]

        if len(np.unique(feats)) == 1:
            feat1 = np.unique(feats)
            self.left = Node(self.input[feats == feat1, :], self.output[feats == feat1], \
                        self.head, self.feat_idx, self.depth + 1, self.max_depth, self)
            self.llable = feat1
            
        else:
            feat1, feat2 = np.unique(feats)
            if self.output[feats == feat1].any():
                self.left = Node(self.input[feats == feat1, :], self.output[feats == feat1], \
                            self.head, self.feat_idx, self.depth + 1, self.max_depth, self)
                self.llable = feat1

            if self.output[feats == feat2].any():
                self.right= Node(self.input[feats == feat2, :], self.output[feats == feat2], \
                            self.head, self.feat_idx, self.depth + 1, self.max_depth, self)
                self.rlable = feat2

        


    
    def train(self):
        '''
        训练决策树
        '''
        
        
        if (len(np.unique(self.output)) == 1) or (self.depth == self.max_depth):
            return
        
        self.splitNode()
<<<<<<< HEAD
=======
        print(self)
>>>>>>> 713d0c82f12244b29158297dca0d1fc5e5f677a4
        if self.left:
            self.left.train()
            
        if self.right:
            self.right.train()

        return
    
    def predictRow(self, X_row):
        '''
        递归当前节点样本
        '''
        if (not self.left) and (not self.right):
            return self.label
        elif X_row[self.feat_idx] == self.llable:
            return self.left.predictRow(X_row)
        elif X_row[self.feat_idx] == self.rlable:
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


def main(train_in, test_in, max_depth, train_out, test_out, metrics):


    # 数据读取
    data, head = data_reading(train_in)
    label = data[:,-1]
    feat = data[:,:-1]
    test_data, _ = data_reading(test_in,)
    test_label = test_data[:,-1]

    # init and training of Decision tree
    root = Node(feat, label, head, 0, 0, max_depth)
    root.train()
    print(root)
    
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
        f.write('error(train): ' + str(trainErr) + '\n')
        f.write('error(test): ' + str(testErr) + '\n')
    #root.treePrint()

    return None




if __name__ == '__main__':
    '''
    运行方法：
<<<<<<< HEAD
    python decisionTree.py ./data/politicians_train.tsv ./data/politicians_test.tsv 6 ./output/pol_6_train.labels ./output/pol_6_test.labels ./output/pol_6_metrics.txt
=======
    python decisionTree.py ./data/education_train.tsv ./data/education_test.tsv 3 ./output/education_3_train.labels ./output/education_3_test.labels ./output/education_3_metrics.txt
>>>>>>> 713d0c82f12244b29158297dca0d1fc5e5f677a4
    '''

    train_in = sys.argv[1]
    test_in = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics = sys.argv[6]
    main(train_in, test_in, max_depth, train_out, test_out, metrics)

