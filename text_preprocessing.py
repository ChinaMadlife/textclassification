import sys
import os
import jieba

#在spyder中编译需要提前配置utf-8输出环境
#reload(sys)
#sys.setdefaultencoding('utf-8')

#设置保存文件函数
def savefile(savepath,content):
	fp=open(savepath,"wb")#二进制写
	fp.write(content)
	fp.close()

#讲文件中内容读取到内存
def readfile(path):
	fp=open(path,"rb")
	content=fp.read()
	fp.close()
	return content

wait_path="C:/Users/user/Desktop/chapter02/train_corpus_small/"# 未分词的语料路径
done_path="C:/Users/user/Desktop/chapter02/done_corpus_small/" # 分词后的语料路径

catelist=os.listdir(wait_path)  #获取未分词路径下的所有子目录

#获取每个目录下的所有文件

for mydir in catelist:
	class_path=wait_path+mydir+"/"             #分类子目录的路径
	done_class_path=done_path+mydir+"/"	   #分词后的类别目录路径
	if not os.path.exists(done_class_path):    #不存在目录则创建
		os.makedirs(done_class_path)
	file_lists=os.listdir(class_path)          #获取class_path下所有目录
	for file_path in file_lists:		   #将每个txt文件读取，分词，再保存
		fullname=class_path+file_path
		content=readfile(fullname)
		content=content.strip().replace(" ","").replace("\r","").replace(" ","")
		content_seg=jieba.cut(content)
		savefile(done_class_path+file_path," ".join(content_seg))

print"中文语料分词结束"
		