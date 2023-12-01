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
fc1_Accuracy = []
fc1_Precision = []
fc1_Recall = []
fc1_F1 = []
fc2_Accuracy = []
fc2_Precision = []
fc2_Recall = []
fc2_F1 = []
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

        print('读取FC1层SVM模型，进行预测和交叉验证')
        dt = joblib.load(svm_model_baseDir+save_svm_model_name[model_index]+'_fc1.pkl')

        print("模型{} FC1层交叉验证:".format(m))
        _accuracy = cross_val_score(dt, flatten_output1, labels, cv=10, scoring='accuracy').mean()
        _precision = cross_val_score(dt, flatten_output1, labels, cv=10, scoring='precision_weighted').mean()
        _recall = cross_val_score(dt, flatten_output1, labels, cv=10, scoring='recall_weighted').mean()
        _f1 = cross_val_score(dt, flatten_output1, labels, cv=10, scoring='f1_weighted').mean()
        print("精确度指标:", _accuracy)
        print("查准率指标:", _precision)
        print("召回率指标:", _recall)
        print("f1得分指标:", _f1)
        fc1_Accuracy.append(round(_accuracy*100,2))
        fc1_Precision.append(round(_precision*100,2))
        fc1_Recall.append(round(_recall*100,2))
        fc1_F1.append(round(_f1*100,2))
        print('\r\n')

        print('读取FC2层SVM模型，进行预测和交叉验证')
        dt = joblib.load(svm_model_baseDir+save_svm_model_name[model_index]+'_fc2.pkl')

        print("模型{} FC2层交叉验证:".format(m))
        _accuracy = cross_val_score(dt, flatten_output2, labels, cv=10, scoring='accuracy').mean()
        _precision = cross_val_score(dt, flatten_output2, labels, cv=10, scoring='precision_weighted').mean()
        _recall = cross_val_score(dt, flatten_output2, labels, cv=10, scoring='recall_weighted').mean()
        _f1 = cross_val_score(dt, flatten_output2, labels, cv=10, scoring='f1_weighted').mean()
        print("精确度指标:", _accuracy)
        print("查准率指标:",_precision )
        print("召回率指标:", _recall)
        print("f1得分指标:", _f1)
        fc2_Accuracy.append(round(_accuracy*100,2))
        fc2_Precision.append(round(_precision*100,2))
        fc2_Recall.append(round(_recall*100,2))
        fc2_F1.append(round(_f1*100,2))
    else:
        print('模型只有FC1层')


        representation_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(all_model_layerName[model_index][0]).output)
        flatten_output1 = representation_model.predict(images)

        print('读取FC1层SVM模型，进行预测和交叉验证')
        dt = joblib.load(svm_model_baseDir+save_svm_model_name[model_index]+'_fc1.pkl')

        print("模型{} FC1层交叉验证:".format(m))
        _accuracy = cross_val_score(dt, flatten_output1, labels, cv=10, scoring='accuracy').mean()
        
        _recall = cross_val_score(dt, flatten_output1, labels, cv=10, scoring='recall_weighted').mean()
        _f1 = cross_val_score(dt, flatten_output1, labels, cv=10, scoring='f1_weighted').mean()
        print("精确度指标:", _accuracy)
        print("查准率指标:", _precision)
        print("召回率指标:", _recall)
        print("f1得分指标:", _f1)
        fc1_Accuracy.append(round(_accuracy*100,2))
        fc1_Precision.append(round(_precision*100,2))
        fc1_Recall.append(round(_recall*100,2))
        fc1_F1.append(round(_f1*100,2))
        print('\r\n')

    model_index += 1

import numpy as np
np.save('weight/pigCoughThermalv2Cnn/lab2_fc1_Accuracy_result.npy', fc1_Accuracy)
np.save('weight/pigCoughThermalv2Cnn/lab2_fc1_Precision_result.npy', fc1_Precision)
np.save('weight/pigCoughThermalv2Cnn/lab2_fc1_Recall_result.npy', fc1_Recall)
np.save('weight/pigCoughThermalv2Cnn/lab2_fc1_F1_result.npy', fc1_F1)

np.save('weight/pigCoughThermalv2Cnn/lab2_fc2_Accuracy_result.npy', fc2_Accuracy)
np.save('weight/pigCoughThermalv2Cnn/lab2_fc2_Precision_result.npy', fc2_Precision)
np.save('weight/pigCoughThermalv2Cnn/lab2_fc2_Recall_result.npy', fc2_Recall)
np.save('weight/pigCoughThermalv2Cnn/lab2_fc2_F1_result.npy', fc2_F1)