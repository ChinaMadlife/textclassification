# -*- coding: utf-8 -*-

import sys  
import os 
import time
from numpy import *
import numpy as np
from Nbayes_lib import *


# # ����utf-8�������
# reload(sys)
# sys.setdefaultencoding('utf-8')

dataSet,listClasses = loadDataSet()
nb = NBayes()
nb.train_set(dataSet,listClasses)
nb.map2vocab(dataSet[3])
print nb.predict(nb.testset)
