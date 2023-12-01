import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve,auc
import os
from sklearn.svm import SVC
from sklearn import metrics
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
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
acoustic = []

file_name = 'data/pigCoughThermalv2/acoustic_data.csv'
acoustic_data = pd.read_csv(file_name)


feat_labels = acoustic_data.columns
del_acoustic_data = []
_importances_features = np.load('weight/pigCoughThermalv2SVM/lab1_importances_features.npy', allow_pickle='TRUE').tolist()
index = 1
for i in _importances_features:
    if i >1:
        del_acoustic_data.append(feat_labels[index])
    index += 1

acoustic_data = acoustic_data.drop(['label'],axis=1)
acoustic_data.set_index('filename',inplace=True)

for f in files:
    file_path = os.path.join(path, str(f)) + '//'
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                train_idx = train_idx + 1
                img_path = os.path.join(file_path, str(file))
                img = image_utils.load_img(img_path,target_size=(100,100))
                img_array = image_utils.img_to_array(img)
                
                temp_file = file.replace('jpg' , 'WAV')
                a_data = acoustic_data.loc[str(temp_file)]
                temp_acoustic_val = []
                for i in a_data:
                    temp_acoustic_val.append(i)

                acoustic.append(temp_acoustic_val)
                images.append(img_array)
                labels.append(fenlei.index(f)) 


images /= 255
print(len(acoustic))


svm_model_baseDir = './weight/pigCoughThermalv2SVM/'
model_filepath='./weight/pigCoughThermalv2Cnn/Diy.cnn.cqt.best.h5'


model = tf.keras.models.load_model(model_filepath)


representation_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('diyfc1').output)
representation_model2 = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('diyfc2').output)
flatten_output1 = representation_model.predict(images)
flatten_output2 = representation_model2.predict(images)

scaler = StandardScaler()

df_1 = pd.DataFrame(flatten_output1)

df_1_scaler = scaler.fit_transform(np.array(df_1, dtype = float))
df_1_pd = pd.DataFrame(df_1_scaler)


df_2 = pd.DataFrame(flatten_output2)

df_2_scaler = scaler.fit_transform(np.array(df_2, dtype = float))
df_2_pd = pd.DataFrame(df_2_scaler)


df_3 = pd.DataFrame(acoustic)

df_3_scaler = scaler.fit_transform(np.array(df_3, dtype = float))
df_3_pd = pd.DataFrame(df_3_scaler)



df_all_pd = pd.concat([df_1_pd, df_3_pd], axis=1, join='outer')
x_train_BestFusion_Acoustic, x_test_BestFusion_Acoustic, y_train_BestFusion_Acoustic, y_test_BestFusion_Acoustic = train_test_split(df_all_pd, labels, test_size=0.3)

dt = SVC()

dt.fit(df_all_pd, labels)
joblib.dump(dt, svm_model_baseDir+'lab3_fc1_acous.pkl')