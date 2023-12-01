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

L = keras.layers
def LeNet_5():
    model = keras.Sequential()
    model.add(L.Conv2D(filters=6,kernel_size=(5,5),padding="valid",activation="tanh",input_shape=[100, 100, 3]))
    model.add(L.MaxPooling2D(pool_size=(2,2)))
    model.add(L.Conv2D(filters=16,kernel_size=(5,5),padding="valid",activation="tanh"))
    model.add(L.MaxPooling2D(pool_size=(2,2)))
    model.add(L.Flatten())
    model.add(L.Dense(120,activation="tanh",name="lenet5fc1"))
    model.add(L.Dense(84,activation="tanh",name="lenet5fc2"))
    model.add(L.Dense(2,activation="softmax"))
    return model

model = LeNet_5()
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model_filepath='./weight/pigCoughThermalv2Cnn/Lenet-5.cnn.cqt.best.h5'

checkpoint = ModelCheckpoint(model_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')
callback_list = [checkpoint]
model.fit(x_train, y_train, epochs=50, validation_split=0.2, callbacks = callback_list)


model = tf.keras.models.load_model(model_filepath)

print('载入最优模型对测试集进行测试')
test_loss, test_acc = model.evaluate(x_test, y_test)
print('test_loss:{}, test_acc{}'.format(test_loss, test_acc))