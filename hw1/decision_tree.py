import numpy as np
from collections import Counter
import argparse

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

def Entropy(Y):
	'''
	信息熵
	'''
	Y_sum = len(Y)
	Y_cnt = list(Counter(Y.tolist()).values())
	Y_etp = sum([- i / Y_sum * np.log2(i / Y_sum) for i in Y_cnt])
	return Y_etp

def loss(Y, X):
	'''
	决策树的负loss函数，希望选取loss最大的特征
	'''
	X_len    = len(X)
	uni_featsm = np.unique(X)
	loss  = Entropy(Y)
	for i in uni_featsm:
		loss  -= (np.count_nonzero(X == i) / X_len) * Entropy(Y[X == i])
	return loss


class Node():
	def __init__(self, X, Y, depth, idx_li, Ncol, maxDepth, father = None):
		self.left   = None
		self.right  = None
		self.father = father
		self.llable = None
		self.rlable = None
		self.X     = X # the X data stored in this node
		self.Y     = Y # the Y data stored in this node
		try:
			self.label = Counter(Y).most_common(1)[0][0] # majority vote
		except:
			self.label = Y[0]
		self.depth = depth
		self.idx   = None # the present column's idx not the absolute idx
		self.Xcol  = idx_li
		self.Ncol  = Ncol # record the names of all columns
		self.idNcol = None # record the split name string
		self.maxDepth = maxDepth
					
	def Choose_Feat(self):
		"""
		选择分类准确率最大的特征
		"""
		mu_li = []
		for i in self.Xcol:
			mu_li.append(loss(self.Y, self.X[:, i]))
		return self.Xcol[mu_li.index(max(mu_li))]


	def splitNode(self):
		'''
		分割特征到下一层节点
		'''
		idx = self.Choose_Feat()
		self.idx = idx
		self.idNcol = self.Ncol[idx]
		feats = self.X[:, idx]
		feat1, feat2 = np.unique(feats)

		newidx_li = self.Xcol.copy()
		newidx_li.remove(idx)

		if self.Y[feats == feat1].any():
			self.left = Node(self.X[feats == feat1, :], self.Y[feats == feat1], \
						self.depth + 1, newidx_li, self.Ncol, self.maxDepth, self)
			self.llable = feat1

		if self.Y[feats == feat2].any():
			self.right= Node(self.X[feats == feat2, :], self.Y[feats == feat2], \
						self.depth + 1, newidx_li, self.Ncol, self.maxDepth, self)
			self.rlable = feat2


	
	def train(self):
		'''
		训练决策树
		'''
		if not(self.Xcol):
			print('single layer decision tree!')
			return
		if (len(np.unique(self.Y)) == 1) or (self.depth == self.maxDepth):
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
		elif X_row[self.idx] == self.llable:
			return self.left.predictRow(X_row)
		elif X_row[self.idx] == self.rlable:
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
	
	print(len(feat[0]))


	# decisionstump 是depth=1 的decisiontree， 初始化
	root = Node(feat, label, 1, [i for i in range(len(feat[0]))], \
				head, int(split_idx) + 1)
	root.train()
	
	# 通过训练好的decision tree 进行推断
	trainOutput = root.predict(data)
	testOutput  = root.predict(test_data)
	
	# 计算预测准确率
	trainErr = np.sum(trainOutput != label) / len(trainOutput)
	testErr = np.sum(testOutput != test_label) / len(testOutput)
	
	with open(train_out, 'w') as nof:
		for i in trainOutput:   
			nof.write(str(i) + '\n')
	with open(test_out, 'w') as sof:
		for i in testOutput:   
			sof.write(str(i) + '\n')
	with open(metrics, 'w') as mof:
		mof.write('error(train): ' + str(trainErr) + '\n')
		mof.write('error(test): ' + str(testErr) + '\n')
	#root.treePrint()

	return None




if __name__ == '__main__':
	'''
	用argparse方法方便调试，用默认命令行参数代替每次输入，调试完成后删去本内容，改用之后sys.argv方法
	运行方法：
	python decisionStump.py
	'''
	args = argparse.ArgumentParser(description='Decision Stump')
	args.add_argument('--train_in', default='./data/small_train.tsv', type=str,
					help='path to the training input .tsv file')
	args.add_argument('--test_in', default='./data/small_test.tsv', type=str,
					help='path to the test input .tsv file')
	args.add_argument('--split_idx', default=3, type=int,
					help='the index of feature at which we split the dataset')
	args.add_argument('--train_out', default='./output/small_3_train.labels', type=str,
					help='path of output .labels file on training data')
	args.add_argument('--test_out', default='./output/small_3_test.labels', type=str,
					help='path of output .labels file on test data')
	args.add_argument('--metrics', default='./output/small_3_metrics.txt', type=str,
					help='path of the output .txt file to metrics')
	
	args = args.parse_args()
	main(args.train_in, args.test_in, args.split_idx, args.train_out, args.test_out, args.metrics)

	'''
	
	调试完成以后main部分改成以下内容
	运行方法：
	python decisionStump.py ./data/politicians_train.tsv ./data/politicians_test.tsv 0 ./output/pol_0_train.labels ./output/pol_0_test.labels ./output/pol_0_metrics.txt
	
	train_in = sys.argv[1]
	test_in = sys.argv[2]
	split_idx = sys.argv[3]
	train_out = sys.argv[4]
	test_out = sys.argv[5]
	metrics = sys.argv[6]
	main(train_in, test_in, split_idx, train_out, test_out, metrics)
	
	'''

