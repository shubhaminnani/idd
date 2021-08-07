#Import Packages
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from deeplab_xception_w import DeepLabV3Plus
import tensorflow as tf
import cv2
from tqdm import tqdm
import os
from glob import glob
import pickle
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.applications.xception import preprocess_input

#Inference Image Size required for Online Evaluation Server
h, w = 720,1280
model = DeepLabV3Plus(h, w, 26)
model.load_weights('top_weights.h5') ##Load top result weights

#Define the inference pipeline
def pipeline(image, fname='', folder=''):
    alpha = 0.5
    dims = image.shape
    image = cv2.resize(image, (1280,720))
    x = image.copy()
    z = model.predict(preprocess_input(np.expand_dims(x, axis=0)))
    z = np.squeeze(z)
    y = np.argmax(z, axis=2)
    y = np.uint8(y)
    y = cv2.resize(y, (dims[1],dims[0]))
    return cv2.imwrite(f'outputs/{folder}/{fname}.png', y)

image_dir = 'dataset/test'
image_list = os.listdir(image_dir)
image_list.sort()
print(f'{len(image_list)} frames found')


for image_dir in image_list:
    os.mkdir(f'outputs/{image_dir}')
    image_list = os.listdir(image_dir)
    image_list.sort()
    print(f'{len(image_list)} frames found')
    for i in tqdm(range(len(image_list))):
        try:
            test = load_img(f'{image_dir}/{image_list[i]}')
            test = img_to_array(test)
            fname = f'{image_list[i]}'
            fname=fname.rsplit('.', 1)[0]
            segmap = pipeline(test, fname=fname, folder=image_dir)
            if segmap == False:
                break
        except Exception as e:
            print(str(e))

