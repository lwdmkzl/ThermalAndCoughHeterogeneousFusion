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


images /= 255


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


df_all_pd = pd.concat([df_1_pd, df_2_pd], axis=1, join='outer')

dt = joblib.load(svm_model_baseDir+'lab3_fc1_fc2.pkl')

print("交叉验证:")
_accuracy = cross_val_score(dt, df_all_pd, labels, cv=10, scoring='accuracy').mean()
_precision = cross_val_score(dt, df_all_pd, labels, cv=10, scoring='precision_weighted').mean()
_recall = cross_val_score(dt, df_all_pd, labels, cv=10, scoring='recall_weighted').mean()
_f1 = cross_val_score(dt, df_all_pd, labels, cv=10, scoring='f1_weighted').mean()
print("精确度指标:", _accuracy)
print("查准率指标:", _precision)
print("召回率指标:", _recall)
print("f1得分指标:", _f1)



np.save('weight/pigCoughThermalv2Cnn/lab_ronghe_lab3.npy', [round(_accuracy*100,2),round(_precision*100,2),round(_recall*100,2),round(_f1*100,2)])