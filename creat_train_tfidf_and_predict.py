import sys  
import os 
#引入Bunch类
from sklearn.datasets.base import Bunch
#引入持久化类
import cPickle as pickle
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.naive_bayes import MultinomialNB

# 配置utf-8输出环境
#reload(sys)
#sys.setdefaultencoding('utf-8')

# 读取文件
def readfile(path):
	fp = open(path,"rb")
	content = fp.read()
	fp.close()
	return content
		
#计算训练语料的tfidf权值并持久化为词袋

#读取bunch对象
def readbunchobj(path):
	file_obj = open(path, "rb")
	bunch = pickle.load(file_obj) 
	file_obj.close()
	return bunch
#写入bunch对象	
def writebunchobj(path,bunchobj):
	file_obj = open(path, "wb")
	pickle.dump(bunchobj,file_obj) 
	file_obj.close()	


#读取停用词表
stopword_path="C:/Users/user/Desktop/chapter02/train_word_bag/hlt_stop_words.txt"
stpwrdlst=readfile(stopword_path).splitlines()


#导入分词后的bunchd对象
path="C:/Users/user/Desktop/chapter02/bunch_data/bunch_set.dat"
bunch=readbunchobj(path)

#构建tf-idf词向量对象
tfidfspace=Bunch(target_name=bunch.target_name,label=bunch.label,filenames=bunch.filenames,tdm=[],vocabulary={})


#构建tf-idf矩阵，并保存字典文件
vectorizer=TfidfVectorizer(stop_words=stpwrdlst,sublinear_tf=True,max_df=0.5)

X_train_counts=vectorizer.fit_transform(bunch.contents)
tfidfspace.vocabulary=vectorizer.vocabulary_

test=readfile("C:/Users/user/Desktop/chapter02/done_test_small/sports/11.txt").splitlines()
#11这个文本是我单独从网上下载的并分词，并没有在训练集里，然而结果分类是sports，还是很成功的
x_new_counts=vectorizer.transform(test)  #此处用transform而不用fit_transform可以共享字典，保证x_new_counts和X_train_counts的列数一样

clf=MultinomialNB(alpha=0.001).fit(X_train_counts,bunch.label)  #alpha取0.001来进行平滑
predicted=clf.predict(x_new_counts)  #进行分类预测

print predicted





