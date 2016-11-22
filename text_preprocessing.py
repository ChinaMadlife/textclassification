import sys
import os
import jieba

#��spyder�б�����Ҫ��ǰ����utf-8�������
#reload(sys)
#sys.setdefaultencoding('utf-8')

#���ñ����ļ�����
def savefile(savepath,content):
	fp=open(savepath,"wb")#������д
	fp.write(content)
	fp.close()

#���ļ������ݶ�ȡ���ڴ�
def readfile(path):
	fp=open(path,"rb")
	content=fp.read()
	fp.close()
	return content

wait_path="C:/Users/user/Desktop/chapter02/train_corpus_small/"# δ�ִʵ�����·��
done_path="C:/Users/user/Desktop/chapter02/done_corpus_small/" # �ִʺ������·��

catelist=os.listdir(wait_path)  #��ȡδ�ִ�·���µ�������Ŀ¼

#��ȡÿ��Ŀ¼�µ������ļ�

for mydir in catelist:
	class_path=wait_path+mydir+"/"             #������Ŀ¼��·��
	done_class_path=done_path+mydir+"/"	   #�ִʺ�����Ŀ¼·��
	if not os.path.exists(done_class_path):    #������Ŀ¼�򴴽�
		os.makedirs(done_class_path)
	file_lists=os.listdir(class_path)          #��ȡclass_path������Ŀ¼
	for file_path in file_lists:		   #��ÿ��txt�ļ���ȡ���ִʣ��ٱ���
		fullname=class_path+file_path
		content=readfile(fullname)
		content=content.strip().replace(" ","").replace("\r","").replace(" ","")
		content_seg=jieba.cut(content)
		savefile(done_class_path+file_path," ".join(content_seg))

print"�������Ϸִʽ���"
		