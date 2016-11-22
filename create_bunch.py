import sys
import os
import cPickle as pickle
from sklearn.datasets.base import Bunch

#配置utf-8输出环境
#reload(sys)
#sys.setdefaultencoding('utf-8')



#保存至文件
def savefile(savepath,content):
	fp = open(savepath,"wb")
	fp.write(content)
	fp.close()
	
# 读取文件	
def readfile(path):
	fp = open(path,"rb")
	content = fp.read()
	fp.close()
	return content

#Bunch类提供一种key,value的对象形式

bunch=Bunch(target_name=[],label=[],filenames=[],contents=[])

seg_path="C:/Users/user/Desktop/chapter02/done_corpus_small/"      #已经分好词的文档
done_path="C:/Users/user/Desktop/chapter02/bunch_data/bunch_set.dat"  #生产的bunch类存放的地方

catelist=os.listdir(seg_path)      #获取分好词的所有子目录
bunch.target_name.extend(catelist) #将子目录存入bunch.target_name

for mydir in catelist:
	class_path=seg_path+mydir+"/"
	file_list=os.listdir(class_path)
	for file_path in file_list:
		fullname=class_path+file_path
		bunch.label.append(mydir)
		bunch.filenames.append(fullname)
		bunch.contents.append(readfile(fullname).strip())  #将文件内容存入bunch.contents

file_obj=open(done_path,"wb")
pickle.dump(bunch,file_obj)
file_obj.close()

print "构建文本对象结束"




