import sys
import os
import cPickle as pickle
from sklearn.datasets.base import Bunch

#����utf-8�������
#reload(sys)
#sys.setdefaultencoding('utf-8')



#�������ļ�
def savefile(savepath,content):
	fp = open(savepath,"wb")
	fp.write(content)
	fp.close()
	
# ��ȡ�ļ�	
def readfile(path):
	fp = open(path,"rb")
	content = fp.read()
	fp.close()
	return content

#Bunch���ṩһ��key,value�Ķ�����ʽ

bunch=Bunch(target_name=[],label=[],filenames=[],contents=[])

seg_path="C:/Users/user/Desktop/chapter02/done_corpus_small/"      #�Ѿ��ֺôʵ��ĵ�
done_path="C:/Users/user/Desktop/chapter02/bunch_data/bunch_set.dat"  #������bunch���ŵĵط�

catelist=os.listdir(seg_path)      #��ȡ�ֺôʵ�������Ŀ¼
bunch.target_name.extend(catelist) #����Ŀ¼����bunch.target_name

for mydir in catelist:
	class_path=seg_path+mydir+"/"
	file_list=os.listdir(class_path)
	for file_path in file_list:
		fullname=class_path+file_path
		bunch.label.append(mydir)
		bunch.filenames.append(fullname)
		bunch.contents.append(readfile(fullname).strip())  #���ļ����ݴ���bunch.contents

file_obj=open(done_path,"wb")
pickle.dump(bunch,file_obj)
file_obj.close()

print "�����ı��������"




