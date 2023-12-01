
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.models import load_model
import tensorflow as tf
tf.compat.v1.disable_eager_execution()



def Grad_Cam(input_model, x, layer_name):

    X = np.expand_dims(x, axis=0)

    X = X.astype('float32')
    preprocessed_input = X / 255.0

    predictions = model.predict(preprocessed_input)
    class_idx = np.argmax(predictions[0])
    class_output = model.output[:, class_idx]

    conv_output = model.get_layer(layer_name).output
    grads = K.gradients(class_output, conv_output)[0]
    gradient_function = K.function([model.input], [conv_output, grads]) 

    output, grads_val = gradient_function([preprocessed_input])
    output, grads_val = output[0], grads_val[0]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)


    cam = cv2.resize(cam, (100, 100), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)
    jetcam = (np.float32(jetcam) + x / 2)

    return jetcam

import os
import matplotlib.pyplot as plt
images_array = []
path = './data/pigCoughThermal/hotpig_gradcam'
files = os.listdir(path)
for f in files:
    img_path = os.path.join(path, str(f))
    images_array.append(img_to_array(load_img(img_path, target_size=(100,100))))

model = load_model("./weight/pigCoughThermalCnn/Diy2.cnn.cqt.best.h5")
image_cam = []
fig, axs = plt.subplots(1,8, figsize=(15, 9))
col = 0
for i in images_array:
    cam = Grad_Cam(model, i, 'max_pooling2d_23')
    image_cam.append(cam)
    axs[col].imshow(array_to_img(i))
    col += 1

plt.show()

col = 0
fig2, axs2 = plt.subplots(1,8, figsize=(15, 9))
for i in image_cam:
    axs2[col].imshow(array_to_img(i))
    col += 1

plt.show()