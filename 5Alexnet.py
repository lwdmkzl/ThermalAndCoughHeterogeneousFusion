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
for f in files:
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

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras

def AlexNet(input_shape=(100,100,3),output_shape=2):

    L = keras.layers
    model = keras.Sequential()


    model.add(
        L.Conv2D(
            filters=48, 
            kernel_size=(11,11),
            strides=(4,4),
            padding='valid',
            input_shape=input_shape,
            activation='relu'
        )
    )
    
    model.add(L.BatchNormalization())


    model.add(
        L.MaxPooling2D(
            pool_size=(3,3), 
            strides=(2,2), 
            padding='valid'
        )
    )


    model.add(
        L.Conv2D(
            filters=128, 
            kernel_size=(5,5), 
            strides=(1,1), 
            padding='same',
            activation='relu'
        )
    )
    
    model.add(L.BatchNormalization())


    model.add(
        L.MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
    )


    model.add(
        L.Conv2D(
            filters=192, 
            kernel_size=(3,3),
            strides=(1,1), 
            padding='same', 
            activation='relu'
        )
    ) 


    model.add(
        L.Conv2D(
            filters=192, 
            kernel_size=(3,3), 
            strides=(1,1), 
            padding='same', 
            activation='relu'
        )
    )


    model.add(
        L.Conv2D(
            filters=128, 
            kernel_size=(3,3), 
            strides=(1,1), 
            padding='same', 
            activation='relu'
        )
    )


    model.add(
        L.MaxPooling2D(
            pool_size=(3,3), 
            strides=(2,2), 
            padding='valid'
        )
    )


    model.add(L.Flatten())
    model.add(L.Dense(1024, activation='relu', name='alexnetfc1'))
    model.add(L.Dropout(0.25))
    
    model.add(L.Dense(1024, activation='relu', name='alexnetfc2'))
    model.add(L.Dropout(0.25))
    
    model.add(L.Dense(output_shape, activation='softmax'))

    return model

model = AlexNet()
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



model_filepath='./weight/pigCoughThermalv2Cnn/AlexNet.cnn.cqt.best.h5'

checkpoint = ModelCheckpoint(model_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')
callback_list = [checkpoint]
model.fit(x_train, y_train, epochs=50, validation_split=0.2, callbacks = callback_list)


model = tf.keras.models.load_model(model_filepath)

print('载入最优模型对测试集进行测试')
test_loss, test_acc = model.evaluate(x_test, y_test)
print('test_loss:{}, test_acc{}'.format(test_loss, test_acc))