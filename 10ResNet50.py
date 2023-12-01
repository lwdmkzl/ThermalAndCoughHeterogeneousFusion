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

from tensorflow.keras.layers import *
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
import os
def conv2d_bn(input, kernel_num, kernel_size=3, strides=1, layer_name='', padding_mode='same'):
    conv1 = Conv2D(kernel_num, kernel_size, strides=strides, padding=padding_mode, name=layer_name + '_conv1')(input)
    batch1 = BatchNormalization(name=layer_name + '_bn1')(conv1)
    return batch1

def shortcut(fx, x, padding_mode='same', layer_name=''):
    layer_name += '_shortcut'
    if x.shape[-1] != fx.shape[-1]:
        k = fx.shape[-1]
        k = int(k)
        identity = conv2d_bn(x, kernel_num=k, kernel_size=1, padding_mode=padding_mode, layer_name=layer_name)
    else:
        identity = x
    return Add(name=layer_name + '_add')([identity, fx])

def bottleneck(input, kernel_num, strides=1, layer_name='bottleneck', padding_mode='same'):
    k1, k2, k3 = kernel_num
    conv1 = conv2d_bn(input, kernel_num=k1, kernel_size=1, strides=strides, padding_mode=padding_mode, layer_name=layer_name+'_1')
    relu1 = ReLU(name=layer_name + '_relu1')(conv1)
    conv2 = conv2d_bn(relu1, kernel_num=k2, kernel_size=3, strides=strides, padding_mode=padding_mode, layer_name=layer_name+'_2')
    relu2 = ReLU(name=layer_name + '_relu2')(conv2)
    conv3 = conv2d_bn(relu2, kernel_num=k3, kernel_size=1, strides=strides, padding_mode=padding_mode, layer_name=layer_name+'_3')

    shortcut_add = shortcut(fx=conv3, x=input, layer_name=layer_name)
    relu3 = ReLU(name=layer_name + '_relu3')(shortcut_add)
 
    return relu3

def basic_block(input, kernel_num=64, strides=1, layer_name='basic', padding_mode='same'):

    conv1 = conv2d_bn(input, kernel_num=kernel_num, strides=strides, kernel_size=3,
                      layer_name=layer_name+'_1', padding_mode=padding_mode)
    relu1 = ReLU(name=layer_name + '_relu1')(conv1)
    conv2 = conv2d_bn(relu1, kernel_num=kernel_num, strides=strides, kernel_size=3,
                      layer_name=layer_name+'_2', padding_mode=padding_mode)
    relu2 = ReLU(name=layer_name + '_relu2')(conv2)
 
    shortcut_add = shortcut(fx=relu2, x=input, layer_name=layer_name)
    relu3 = ReLU(name=layer_name + '_relu3')(shortcut_add)
    return relu3

def make_layer(input, block, block_num, kernel_num, layer_name=''):
        x = input
        for i in range(1, block_num+1):
            x = block(x, kernel_num=kernel_num, strides=1, layer_name=layer_name+str(i), padding_mode='same')
        return x

def ResNet(input_shape, nclass, net_name='resnet18'):
    """
        :param input_shape:
        :param nclass:
        :param block:
        :return:
    """
    block_setting = {}
    block_setting['resnet18'] = {'block': basic_block, 'block_num': [2, 2, 2, 2], 'kernel_num': [64, 128, 256, 512]}
    block_setting['resnet34'] = {'block': basic_block, 'block_num': [3, 4, 6, 3], 'kernel_num': [64, 128, 256, 512]}
    block_setting['resnet50'] = {'block': bottleneck, 'block_num': [3, 4, 6, 3], 'kernel_num': [[64, 64, 256], [128, 128, 512],
                                                                           [256, 256, 1024], [512, 512, 2048]]}
    block_setting['resnet101'] = {'block': bottleneck, 'block_num': [3, 4, 23, 3], 'kernel_num': [[64, 64, 256], [128, 128, 512],
                                                                           [256, 256, 1024], [512, 512, 2048]]}
    block_setting['resnet152'] = {'block': bottleneck, 'block_num': [3, 8, 36, 3], 'kernel_num': [[64, 64, 256], [128, 128, 512],
                                                                           [256, 256, 1024], [512, 512, 2048]]}
    net_name = 'resnet18' if not block_setting.__contains__(net_name) else net_name
    block_num = block_setting[net_name]['block_num']
    kernel_num = block_setting[net_name]['kernel_num']
    block = block_setting[net_name]['block']
 
    input_ = Input(shape=input_shape)
    conv1 = conv2d_bn(input_, 64, kernel_size=7, strides=2, layer_name='first_conv')
    pool1 = MaxPool2D(pool_size=(3, 3), strides=2, padding='same', name='pool1')(conv1)
 
    conv = pool1
    for i in range(4):
          conv = make_layer(conv, block=block, block_num=block_num[i], kernel_num=kernel_num[i], layer_name='layer'+str(i+1))
 
    pool2 = GlobalAvgPool2D(name='globalavgpool')(conv)

 
    model = Model(inputs=input_, outputs=pool2, name='ResNet18')
    
 
    return model

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
conv_base = ResNet(input_shape=(100, 100, 3), nclass=2, net_name='resnet50')

model = models.Sequential()
model.add(conv_base)

model.add(layers.Dense(256, activation='relu', name='resnet50fc1'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2,activation='softmax'))

model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model_filepath='./weight/pigCoughThermalv2Cnn/ResNet50.cnn.cqt.best.h5'

checkpoint = ModelCheckpoint(model_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')
callback_list = [checkpoint]
model.fit(x_train, y_train, batch_size=32, epochs=50, validation_split=0.2, callbacks = callback_list)


model = tf.keras.models.load_model(model_filepath)

print('载入最优模型对测试集进行测试')
test_loss, test_acc = model.evaluate(x_test, y_test)
print('test_loss:{}, test_acc{}'.format(test_loss, test_acc))