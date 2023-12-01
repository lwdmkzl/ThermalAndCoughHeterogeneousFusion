import pandas as pd
import os
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import cross_val_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

svm_model_baseDir = './weight/pigCoughThermalv2SVM/'
model_baseDir = './weight/pigCoughThermalv2Cnn/'


all_model = ['Lenet-5.cnn.cqt.best.h5','AlexNet.cnn.cqt.best.h5','DenseNet121.cnn.cqt.best.h5','Diy.cnn.cqt.best.h5','Vgg16.cnn.cqt.best.h5','Vgg19.cnn.cqt.best.h5','ResNet50.cnn.cqt.best.h5','ResNet101.cnn.cqt.best.h5','ResNet152.cnn.cqt.best.h5',]
all_model_name = ['Lenet-5','AlexNet','DenseNet121','自定义网络','Vgg16','Vgg19','ResNet50','ResNet101','ResNet152']

all_model_layerName = [['lenet5fc1','lenet5fc2'],['alexnetfc1','alexnetfc2'],['densenet121fc1'],['diyfc1','diyfc2'],['vgg16fc1','vgg16fc2'],['vgg19fc1','vgg19fc2'],['resnet50fc1'],['resnet101fc1'],['resnet152fc1']]

save_svm_model_name = ['lab2_Lenet5','lab2_AlexNet','lab2_DenseNet121','lab2_DiyNet','lab2_Vgg16','lab2_Vgg19','lab2_ResNet50','lab2_ResNet101','lab2_ResNet152']

model_index = 0
for m in all_model:
    print("**********************{}****************************".format(m))
    model_filepath = os.path.join(model_baseDir,m)
    print('载入模型',model_filepath)
    print('抽取全连接层：')
    print(all_model_layerName[model_index])

    model = tf.keras.models.load_model(model_filepath)

    if len(all_model_layerName[model_index])==2:
        print('模型有FC1、FC2层')

        representation_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(all_model_layerName[model_index][0]).output)
        flatten_output1 = representation_model.predict(images)

        representation_model2 = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(all_model_layerName[model_index][1]).output)
        flatten_output2 = representation_model2.predict(images)
        print('FC1层SVM模型训练')
        dt = SVC()
        dt.fit(flatten_output1, labels)
        joblib.dump(dt, svm_model_baseDir+save_svm_model_name[model_index]+'_fc1.pkl')

        print('FC2层SVM模型训练')
        dt2 = SVC()
        dt2.fit(flatten_output2, labels)
        joblib.dump(dt2, svm_model_baseDir+save_svm_model_name[model_index]+'_fc2.pkl')
    else:
        print('模型只有FC1层')

        representation_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(all_model_layerName[model_index][0]).output)
        flatten_output1 = representation_model.predict(images)
        
        print('FC1层SVM模型训练')
        dt = SVC()
        dt.fit(flatten_output1, labels)
        joblib.dump(dt, svm_model_baseDir+save_svm_model_name[model_index]+'_fc1.pkl')
    
    model_index += 1