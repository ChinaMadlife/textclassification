import sys  
import os 
#����Bunch��
from sklearn.datasets.base import Bunch
#����־û���
import cPickle as pickle
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import TfidfVectorizer  

# ����utf-8�������
#reload(sys)
#sys.setdefaultencoding('utf-8')

# ��ȡ�ļ�
def readfile(path):
	fp = open(path,"rb")
	content = fp.read()
	fp.close()
	return content
		
#����ѵ�����ϵ�tfidfȨֵ���־û�Ϊ�ʴ�

#��ȡbunch����
def readbunchobj(path):
	file_obj = open(path, "rb")
	bunch = pickle.load(file_obj) 
	file_obj.close()
	return bunch
#д��bunch����	
def writebunchobj(path,bunchobj):
	file_obj = open(path, "wb")
	pickle.dump(bunchobj,file_obj) 
	file_obj.close()	


#��ȡͣ�ôʱ�
stopword_path="C:/Users/user/Desktop/chapter02/train_word_bag/hlt_stop_words.txt"
stpwrdlst=readfile(stopword_path).splitlines()


#����ִʺ��bunchd����
path="C:/Users/user/Desktop/chapter02/bunch_data/bunch_set.dat"
bunch=readbunchobj(path)

#����tf-idf����������
tfidfspace=Bunch(target_name=bunch.target_name,label=bunch.label,filenames=bunch.filenames,tdm=[],vocabulary={})


#����tf-idf���󣬲������ֵ��ļ�
vectorizer=TfidfVectorizer(stop_words=stpwrdlst,sublinear_tf=True,max_df=0.5)

X_train_counts=vectorizer.fit_transform(bunch.contents)
tf_transformer=TfidfTransformer().fit(X_train_counts)
tfidfspace.tdm=tf_transformer.transform(X_train_counts)
tfidfspace.vocabulary=vectorizer.vocabulary_


#�����ʴ��־û�
space_path="C:/Users/user/Desktop/chapter02/train_word_bag/tfdifspace.dat"
writebunchobj(space_path,tfidfspace)

print "tf-idf�������ռ䴴���ɹ�"










