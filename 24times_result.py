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


all_model = ['Diy.cnn.cqt.best.h5','Lenet-5.cnn.cqt.best.h5','AlexNet.cnn.cqt.best.h5','DenseNet121.cnn.cqt.best.h5','Vgg16.cnn.cqt.best.h5','Vgg19.cnn.cqt.best.h5','ResNet50.cnn.cqt.best.h5','ResNet101.cnn.cqt.best.h5','ResNet152.cnn.cqt.best.h5',]

all_model_layerName = ['diyfc1','lenet5fc1','alexnetfc1','densenet121fc1','vgg16fc1','vgg19fc1','resnet50fc1','resnet101fc1','resnet152fc1']

save_svm_model_name = ['lab3_fc1_acous','lab4_Lenet5_time','lab4_AlexNet_time','lab4_DenseNet121_time','lab4_Vgg16_time','lab4_Vgg19_time','lab4_ResNet50_time','lab4_ResNet101_time','lab4_ResNet152_time']

all_model_name = ['diy','Lenet5','AlexNet','DenseNet121','Vgg16','Vgg19','ResNet50','ResNet101','ResNet152']



all_model = ['ResNet50.cnn.cqt.best.h5','ResNet101.cnn.cqt.best.h5','ResNet152.cnn.cqt.best.h5']

all_model_layerName = ['resnet50fc1','resnet101fc1','resnet152fc1']

save_svm_model_name = ['lab4_ResNet50_time','lab4_ResNet101_time','lab4_ResNet152_time']

all_model_name = ['ResNet50','ResNet101','ResNet152']



model_index = 0

scaler = StandardScaler()

for m in all_model:
    print("**********************{}****************************".format(m))
    model_filepath = os.path.join(model_baseDir,m)
    print('载入模型',model_filepath)
    print(all_model_layerName[model_index])
    
    
    _lst = []
    _lst_temp = []
    for i in range(26):


        model = tf.keras.models.load_model(model_filepath)

        representation_model1 = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(all_model_layerName[model_index]).output)

        dt = joblib.load(svm_model_baseDir+save_svm_model_name[model_index]+'.pkl')
    
        pred_images = []
        new_audio_path = './data/pigCoughThermal/audio/cough/20221110162631_01_kesou.WAV'
        new_img_path = './data/pigCoughThermal/hotpig/cough/20221110162631_01_kesou.jpg'
        load_img = image_utils.load_img(new_img_path,target_size=(100,100))
        load_img_array = image_utils.img_to_array(load_img)
        pred_images.append(load_img_array)

        pred_images /= 255

        pred_acoustic = []

        y, sr = librosa.load(new_audio_path,sr=None, mono=True, duration=30)


        zcr = librosa.feature.zero_crossing_rate(y)

        rmse = librosa.feature.rms(y=y)


        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)

        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)

        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        flatness = librosa.feature.spectral_flatness(y=y)

        o_env = librosa.onset.onset_strength(y=y, sr=sr)


        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        temp_acoustic = [np.mean(chroma_stft),np.mean(chroma_cq),np.mean(rmse),np.mean(spec_cent),np.mean(spec_bw),np.mean(rolloff),np.mean(flatness),np.mean(o_env),np.mean(zcr)]

        for e in mfcc:
            temp_acoustic.append(np.mean(e))
        pred_acoustic.append(temp_acoustic)

        output1 = representation_model1(pred_images,training=False)


        _df_1 = pd.DataFrame(output1)

        _df_1_scaler = scaler.fit_transform(np.array(_df_1, dtype = float))
        _df_1_pd = pd.DataFrame(_df_1_scaler)


        _df_3 = pd.DataFrame(pred_acoustic)

        _df_3_scaler = scaler.fit_transform(np.array(_df_3, dtype = float))
        _df_3_pd = pd.DataFrame(_df_3_scaler)


        _df_all_pd = pd.concat([_df_1_pd, _df_3_pd], axis=1, join='outer')

        dt.predict(_df_all_pd)


        if len(_lst_temp)==5:
            _lst.append(sum(_lst_temp)/len(_lst_temp))
            _lst_temp = []
            _lst_temp.append(end_time - start_time)
        else:
            _lst_temp.append(end_time - start_time)
    
    print('每次用时')
    print(_lst)
    average = sum(_lst) / len(_lst)
    print('平均用时')
    print(average)

    _lst.append(average)
    timeDicts[all_model_name[model_index]] = _lst
        
    model_index += 1
    
import numpy as np
np.save('weight/pigCoughThermalCnn/times.npy', timeDicts)