from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR

from matplotlib import font_manager
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import warnings
import tensorflow
import tensorflow.keras
from tensorflow.keras.applications import imagenet_utils
from sklearn.svm import SVC


fenlei = 'cough nocough'.split()
file_name = 'data/pigCoughThermalv2/acoustic_data.csv'

data = pd.read_csv(file_name)
data = data.drop(['filename'],axis=1)
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

svm_model_baseDir = './weight/pigCoughThermalv2SVM/'
feat_labels = data.columns


selector = joblib.load(svm_model_baseDir+'lab1_fold5_f1.pkl')
print('开始5折交叉')
print('每个特征的得分排名，特征得分越低（1最好），表示特征越好')
importances = selector.ranking_
print(importances)
np.save('weight/pigCoughThermalv2SVM/lab1_importances_features.npy', importances)
print('5折交叉自动挑选了{}个特征'.format(selector.n_features_))
indices = np.argsort(importances)



c5 = selector.grid_scores_
c5_mean = []
for i in c5:
    c5_mean.append(i.mean())


selector = joblib.load(svm_model_baseDir+'lab1_fold10_f1.pkl')
print('\n\r开始10折交叉')
print('每个特征的得分排名，特征得分越低（1最好），表示特征越好')
importances = selector.ranking_
print(importances)
print('10折交叉挑选了{}特征'.format(selector.n_features_))
indices = np.argsort(importances)




c10 = selector.grid_scores_
c10_mean = []
for i in c10:
    c10_mean.append(i.mean())


font_path = "./TimesNewRoman.ttf"
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)






plt.figure(figsize=(6, 3))
plt.ylim(ymin=0.7,ymax=1)



ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(1*0.5)
ax.spines['bottom'].set_linewidth(1*0.5)
plt.xlabel("Numbers of features selected")
plt.ylabel("Cross validation F1-score")
plt.title("5-Fold cross-validation")
alpha_size=0.8



markerList = ["*","d","v","^","o"]
labelList = ['val01','val02','val03','val04','val05']
f5_1 = []
f5_2 = []
f5_3 = []
f5_4 = []
f5_5 = []
index = 0
_x = [i for i in range(1, len(c5) + 1)]
for i in c5:
    f5_1.append(i[0])
    f5_2.append(i[1])
    f5_3.append(i[2])
    f5_4.append(i[3])
    f5_5.append(i[4])


plt.scatter(_x, f5_1, s=20, alpha=alpha_size, c=colorList[0], marker=markerList[0], label=labelList[0])
plt.scatter(_x, f5_2, s=13, alpha=alpha_size, c=colorList[1], marker=markerList[1], label=labelList[1])
plt.scatter(_x, f5_3, s=11, alpha=alpha_size, c=colorList[2], marker=markerList[2], label=labelList[2])
plt.scatter(_x, f5_4, s=12, alpha=alpha_size, c=colorList[3], marker=markerList[3], label=labelList[3])
plt.scatter(_x, f5_5, s=13, alpha=alpha_size, c=colorList[4], marker=markerList[4], label=labelList[4])

plt.plot(range(1, len(c5) + 1), c5_mean, marker='s',markersize=3, markerfacecolor=colorMean, markeredgecolor=colorMean, 



max_y = max(c5_mean)
max_x = c5_mean.index(max_y)+1
print(f'\n\n说明1：5折交叉验证使用{max_x}个特征时，以F1为评价标准的f1-score平均得分最高，值为{max_y}')


selector_f1 = joblib.load(svm_model_baseDir+'lab1_fold5_f1.pkl')
selector_acc = joblib.load(svm_model_baseDir+'lab1_fold5_acc.pkl')
selector_prec = joblib.load(svm_model_baseDir+'lab1_fold5_prec.pkl')
selector_recall = joblib.load(svm_model_baseDir+'lab1_fold5_recall.pkl')
print('5折交叉的F1=',round(max_y*100,2),'Accuracy=',round(selector_acc.grid_scores_[max_x-1].mean()*100,2),'Precision=',round(selector_prec.grid_scores_[max_x-1].mean()*100,2),'Recall=',round(selector_recall.grid_scores_[max_x-1].mean()*100,2))
ronghe_lab1 = [round(selector_acc.grid_scores_[max_x-1].mean()*100,2),round(selector_prec.grid_scores_[max_x-1].mean()*100,2),round(selector_recall.grid_scores_[max_x-1].mean()*100,2),round(max_y*100,2)]
np.save('weight/pigCoughThermalv2Cnn/lab_ronghe_lab1.npy', ronghe_lab1)



plt.annotate(f'number of features: {max_x}', xy=(max_x, max_y), xytext=(max_x-5.2, max_y-0.055),



plt.plot([0], [0], label='mean', marker='s',markersize=3.6, markerfacecolor=colorMean, markeredgecolor=colorMean, 

plt.legend(frameon=False,bbox_to_anchor=(1.033,0.6))
plt.savefig('v2_pic6.1.png', dpi = 100, bbox_inches='tight')
plt.show()



plt.figure(figsize=(5, 3))
plt.ylim(ymin=0.7,ymax=1)



ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(1*0.5)
ax.spines['bottom'].set_linewidth(1*0.5)
plt.xlabel("Numbers of features selected")

plt.ylabel("Cross validation F1-score")
plt.title("10-Fold cross-validation")



markerList = ["*","d","v","^","o","<",">","s","p","X"]
labelList = ['val01','val02','val03','val04','val05','val06','val07','val08','val09','val10']
f10_1 = []
f10_2 = []
f10_3 = []
f10_4 = []
f10_5 = []
f10_6 = []
f10_7 = []
f10_8 = []
f10_9 = []
f10_10 = []
index = 0
_x = [i for i in range(1, len(c10) + 1)]
for i in c10:
    f10_1.append(i[0])
    f10_2.append(i[1])
    f10_3.append(i[2])
    f10_4.append(i[3])
    f10_5.append(i[4])
    f10_6.append(i[5])
    f10_7.append(i[6])
    f10_8.append(i[7])
    f10_9.append(i[8])
    f10_10.append(i[9])

plt.scatter(_x, f10_1, s=25, alpha=alpha_size, c=colorList[0], marker=markerList[0], label=labelList[0])
plt.scatter(_x, f10_2, s=15, alpha=alpha_size, c=colorList[1], marker=markerList[1], label=labelList[1])
plt.scatter(_x, f10_3, s=15, alpha=alpha_size, c=colorList[2], marker=markerList[2], label=labelList[2])
plt.scatter(_x, f10_4, s=15, alpha=alpha_size, c=colorList[3], marker=markerList[3], label=labelList[3])
plt.scatter(_x, f10_5, s=15, alpha=alpha_size, c=colorList[4], marker=markerList[4], label=labelList[4])
plt.scatter(_x, f10_6, s=15, alpha=alpha_size, c=colorList[5], marker=markerList[5], label=labelList[5])
plt.scatter(_x, f10_7, s=15, alpha=alpha_size, c=colorList[6], marker=markerList[6], label=labelList[6])
plt.scatter(_x, f10_8, s=13, alpha=alpha_size, c=colorList[7], marker=markerList[7], label=labelList[7])
plt.scatter(_x, f10_9, s=17, alpha=alpha_size, c=colorList[8], marker=markerList[8], label=labelList[8])
plt.scatter(_x, f10_10, s=17, alpha=alpha_size, c=colorList[9], marker=markerList[9], label=labelList[9])

plt.plot(range(1, len(c10) + 1), c10_mean, marker='s',markersize=3, markerfacecolor=colorMean, markeredgecolor=colorMean, 


max_y = max(c10_mean)
max_x = c10_mean.index(max_y)+1


selector_f1 = joblib.load(svm_model_baseDir+'lab1_fold10_f1.pkl')
selector_acc = joblib.load(svm_model_baseDir+'lab1_fold10_acc.pkl')
selector_prec = joblib.load(svm_model_baseDir+'lab1_fold10_prec.pkl')
selector_recall = joblib.load(svm_model_baseDir+'lab1_fold10_recall.pkl')
print(f'\n\n说明2：10折交叉验证使用{max_x}个特征时，以F1为评价标准的f1-score平均得分最高，值为{max_y}')
print('10折交叉的F1=',round(max_y*100,2),'Accuracy=',round(selector_acc.grid_scores_[max_x-1].mean()*100,2),'Precision=',round(selector_prec.grid_scores_[max_x-1].mean()*100,2),'Recall=',round(selector_recall.grid_scores_[max_x-1].mean()*100,2))



plt.annotate(f'number of features: {max_x}', xy=(max_x, max_y), xytext=(max_x-6, max_y-0.055),


plt.plot([0], [0], label='mean', marker='s',markersize=3.6, markerfacecolor=colorMean, markeredgecolor=colorMean, 

plt.legend(frameon=False,bbox_to_anchor=(0.98,1.005))
plt.savefig('v2_pic6.2.png', dpi = 100, bbox_inches='tight')
plt.show()