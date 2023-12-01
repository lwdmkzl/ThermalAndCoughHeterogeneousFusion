import pandas as pd
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import os
from sklearn.svm import SVC
from sklearn import metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve,auc
import joblib   # jbolib模块
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tensorflow.keras.preprocessing import image as image_utils
import random
from sklearn.model_selection import train_test_split
import numpy as np
import os

train_idx = 0
npy_idx = 0
path = './data/pigCoughThermal/hotpig'
files = os.listdir(path)
random.shuffle(files)
images = []
labels = []
for f in files: #目录下所有文件夹
    file_path = os.path.join(path, str(f)) + '/'
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                train_idx = train_idx + 1
                img_path = os.path.join(file_path, str(file))
                img = image_utils.load_img(img_path,target_size=(100,100))
                img_array = image_utils.img_to_array(img)
                images.append(img_array)
                labels.append(fenlei.index(f)) 
images = np.array(images)   #（num, h, w, 3）
labels = np.array(labels)   #(num, )
images /= 255

svm_model_baseDir = './weight/pigCoughThermalv2SVM/'
model_filepath='./weight/pigCoughThermalv2Cnn/Diy.cnn.cqt.best.h5'

# 载入模型
model = tf.keras.models.load_model(model_filepath)

# 获取flatten层的输出
representation_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('diyfc1').output)
representation_model2 = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('diyfc2').output)
flatten_output1 = representation_model.predict(images)
flatten_output2 = representation_model2.predict(images)

scaler = StandardScaler()
#FC1深度特征
df_1 = pd.DataFrame(flatten_output1)
#深度特征归一化
df_1_scaler = scaler.fit_transform(np.array(df_1, dtype = float))
df_1_pd = pd.DataFrame(df_1_scaler)
#FC2深度特征
df_2 = pd.DataFrame(flatten_output2)
#深度特征归一化
df_2_scaler = scaler.fit_transform(np.array(df_2, dtype = float))
df_2_pd = pd.DataFrame(df_2_scaler)

#FC1、FC2深度特征合并
df_all_pd = pd.concat([df_1_pd, df_2_pd], axis=1, join='outer')

dt = SVC()

# 开始训练
dt.fit(df_all_pd, labels)
joblib.dump(dt, svm_model_baseDir+'lab3_fc1_fc2.pkl')