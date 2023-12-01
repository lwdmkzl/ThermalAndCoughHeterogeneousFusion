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

file_name = 'data/pigCoughThermal/acoustic_data.csv'
acoustic_data = pd.read_csv(file_name)
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

svm_model_baseDir = './weight/pigCoughThermalv2SVM/'
model_baseDir = './weight/pigCoughThermalv2Cnn/'

all_model = ['Lenet-5.cnn.cqt.best.h5','AlexNet.cnn.cqt.best.h5','DenseNet121.cnn.cqt.best.h5','Vgg16.cnn.cqt.best.h5','Vgg19.cnn.cqt.best.h5','ResNet50.cnn.cqt.best.h5','ResNet101.cnn.cqt.best.h5','ResNet152.cnn.cqt.best.h5',]

all_model_layerName = ['lenet5fc1','alexnetfc1','densenet121fc1','vgg16fc1','vgg19fc1','resnet50fc1','resnet101fc1','resnet152fc1']

save_svm_model_name = ['lab4_Lenet5','lab4_AlexNet','lab4_DenseNet121','lab4_Vgg16','lab4_Vgg19','lab4_ResNet50','lab4_ResNet101','lab4_ResNet152']



all_model = ['ResNet50.cnn.cqt.best.h5','ResNet101.cnn.cqt.best.h5','ResNet152.cnn.cqt.best.h5',]

all_model_layerName = ['resnet50fc1','resnet101fc1','resnet152fc1']

save_svm_model_name = ['lab4_ResNet50','lab4_ResNet101','lab4_ResNet152']


model_index = 0

scaler = StandardScaler()

for m in all_model:
    print("**********************{}****************************".format(m))
    model_filepath = os.path.join(model_baseDir,m)
    print('载入模型',model_filepath)
    print(all_model_layerName[model_index])

    model = tf.keras.models.load_model(model_filepath)

    representation_model1 = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(all_model_layerName[model_index]).output)
    


    print('训练一个分类器');
    output_train = representation_model1(images,training=False)

    _df_1_train = pd.DataFrame(output_train)

    _df_1_scaler_train = scaler.fit_transform(np.array(_df_1_train, dtype = float))
    _df_1_pd_train = pd.DataFrame(_df_1_scaler_train)

    _df_3_train = pd.DataFrame(acoustic)

    _df_3_scaler_train = scaler.fit_transform(np.array(_df_3_train, dtype = float))
    _df_3_pd_train = pd.DataFrame(_df_3_scaler_train)

    _df_all_pd_train = pd.concat([_df_1_pd_train, _df_3_pd_train], axis=1, join='outer')
    x_train2, x_test2, y_train2, y_test2 = train_test_split(_df_all_pd_train, labels, test_size=0.3)
    dt = SVC()
    dt.fit(x_train2, y_train2)
    
    joblib.dump(dt, svm_model_baseDir+save_svm_model_name[model_index]+'_time.pkl')
    
    model_index += 1