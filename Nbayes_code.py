import sys
import os
import numpy as np

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him','my'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec


class NBayes(object):
	def __init__(self):
		self.vocabulary=[]      #词典（从给定数据集中生成）
		self.idf=0		#词典的逆文本频率
		self.tf=0		#训练集的单词频率
		self.tdm=0		#p(x|yi)
		self.Pcates={}		#字典类型，每个类以及对应的概率，即p(yi)
		self.doclength=0	#训练集的文本个数
		self.vocablen=0	#字典长度
		self.testset=0		#测试集
		self.labels=[]		#文本类别

	def cate_prob(self,classVec):			#计算p(yi)
		self.labels=classVec			#将类别信息导入
		labeltemps=set(self.labels)		#生成集合类（类别不可重复）
		for labeltemp in labeltemps:
			self.Pcates[labeltemp]=float(self.labels.count(labeltemp))/float(len(self.labels))		#Pcates包含每个类以及每个类的概率

	def train_set(self,trainset,classVec):
		self.cate_prob(classVec)
		self.doclength=len(trainset)
		tempset=set()
		[tempset.add(word) for doc in trainset for word in doc]	#生成字典
		self.vocabulary=list(tempset)		#集合类变为列表类
		self.vocablen=len(self.vocabulary)
		self.cacl_tfidf(trainset)
		self.build_tdm()

	def cacl_tfidf(self,trainset):
		self.idf=np.zeros([1,self.vocablen])
		self.tf=np.zeros([self.doclength,self.vocablen])
		for indx in xrange(self.doclength):
			for word in trainset[indx]:
				self.tf[indx,self.vocabulary.index(word)]+=1
			self.tf[indx]=self.tf[indx]/float(len(trainset[indx]))
			for signleword in set(trainset[indx]):
				self.idf[0,self.vocabulary.index(signleword)]+=1	#判断每个词出现多少几个文本中
		self.idf=np.log(float(self.doclength)/self.idf)		#计算逆文本概率
		self.tf=np.multiply(self.tf,self.idf)			#相同列数的矩阵与向量的点乘

	def map2vocab(self,testdata):			#测试集映射到字典，生成词频矩阵
		self.testset=np.zeros([1,self.vocablen])
		for word in testdata:
			self.testset[0,self.vocabulary.index(word)]+=1
	
	def build_tdm(self):		#按分类向量，计算每行（类别）的矩阵
		self.tdm=np.zeros([len(self.Pcates),self.vocablen])
		sumlist=np.zeros([len(self.Pcates),1])
		for indx in xrange(self.doclength):
			self.tdm[self.labels[indx]]+=self.tf[indx]
			sumlist[self.labels[indx]]=np.sum(self.tdm[self.labels[indx]])
		self.tdm=self.tdm/sumlist		#p(x|yi)，计算的是词典中每个词出现的tfidf概率，所以应该先累加，再归一化。统一度量
		self.tdm = (self.tdm+0.001)/(1+0.001*len(self.tdm[0])) #在计算特征时引入Lidstone平滑，但是参数的具体选择还有待商榷。

	def predict(self,testset):
		if np.shape(testset)[1] != self.vocablen:
			print "输入错误"
			exit(0)
		predvalue = 0
		predclass = ""
		for tdm_vect,keyclass in zip(self.tdm,self.Pcates):
			# P(x|yi)P(yi)
			cal=1
			b=testset*tdm_vect*self.Pcates[keyclass]
	
		#在《机器学习》一书中，此处是通过累加进行比较，但是我觉得有点不对，所以用的累乘
			print b
			for a in b[0]:
				if a>0:
					cal=cal*a

			if cal > predvalue:
				predvalue = cal
				predclass = keyclass
		return predclass
