# coding=utf-8
import os
import glob
import numpy as np
import PIL 
import pdb

from model.Adam_lr_mult import Adam_lr_mult
from model.ModelMGPU import *
from model.model import top_3_accuracy

# Keras
import tensorflow as tf
from keras.applications.xception import preprocess_input
from keras.preprocessing import image
from keras.models import load_model
from keras.metrics import top_k_categorical_accuracy
# Flask utils
from flask import Flask, redirect, url_for, request, render_template

# Define a flask app
app = Flask(__name__)

MODEL_PATH = 'model/Xception-fish.17-2.11.hdf5'
# Use Keras pre-trained Xception to classify image
model = load_model(MODEL_PATH, custom_objects={'Adam_lr_mult': Adam_lr_mult, 'top_3_accuracy': top_3_accuracy})
print('Model loaded')

#import predictions decoder
class_dict = np.load('model/decode_class.npy')
class_dict = class_dict.item()

# need to get tensorflow on the same thread as flask
global graph
graph = tf.get_default_graph()


def model_predict(img, model):
    img = image.load_img(img, target_size=(299, 299))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    with graph.as_default():
        preds = model.predict(x)
    return preds

def decode_predictions(preds, top):

    preds = preds[0]
    idxes = (-preds).argsort()
    idx = idxes[:top]

    result = {}
    for i in idx:

    	key = class_dict[i]

    	value = preds[i]
    	result[key] = value
    return result


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Make prediction
        preds = model_predict(f, model)

        # Process your result for human
        pred_class = decode_predictions(preds, top=5) 
        result = str(pred_class)               # Convert to string
    return render_template('predict.html', result=result)


if __name__ == '__main__':
    app.run(host= "0.0.0.0",port=5003, debug=True)