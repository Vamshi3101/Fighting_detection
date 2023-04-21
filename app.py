from flask import Flask, render_template, Response
import os
import webview
import platform
from IPython.display import clear_output
import tensorflow as tf
import cv2
import numpy as np
import pickle
import matplotlib
matplotlib.use("TkAgg")
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Dropout,Flatten,Dense
import argparse
import pickle
import time
from keras.models import load_model
from collections import deque
from PIL import Image
import matplotlib.pyplot as plt
tf.random.set_seed(73)
TPU_INIT = False
epochs = 150

from keras import regularizers
kernel_regularizer = regularizers.l2(0.0001)

from keras.applications import MobileNetV2

def load_layers():
    input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, ColorChannels))
    baseModel = MobileNetV2(pooling='avg',
                            include_top=False, 
                            input_tensor=input_tensor)
    
    headModel = baseModel.output   
    headModel = Dense(1, activation="sigmoid")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)

    for layer in baseModel.layers:
        layer.trainable = False
    model.compile(loss="binary_crossentropy",
                    optimizer='adam',
                    metrics=["accuracy"])

    return model
IMG_SIZE = 128
ColorChannels = 3
if TPU_INIT:
    with tpu_strategy.scope():
        model = load_layers()
else:
    model = load_layers()
model.load_weights('model.h5')
args_model = "model.h5"
sourcePath = 'output'
app = Flask(__name__)
model = load_model('./model.h5')
Q = deque(maxlen=128)
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
IMG_SIZE = 128
def gen_frames():  
    b=0
    while True:
        success, frame = cap.read()  # read the camera frame
        output = frame.copy()
        if not success:
            break
        else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                output = cv2.resize(frame,(128,128)).astype("float32")
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                # frame = cv2.resize(frame, (128, 128)).astype("float32")
                # output = output.reshape(IMG_SIZE, IMG_SIZE, 3) / 255
                preds = model.predict(np.expand_dims(output, axis=0))[0]
                Q.append(preds)

                results = np.array(Q).mean(axis=0)
                i = (preds > .35)[0] #np.argmax(results)

                label = i
                text = "Violence: {}".format(label)
                print('prediction:', text)
                color = (0, 255, 0)
                if label:
                    color = (0,0,255)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (400, 640)).astype("float32")
                frame = cv2.flip(frame,1)
                cv2.putText(frame,str(preds*100),(250,600),cv2.FONT_HERSHEY_SIMPLEX,0.8, color, 2)
                cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,.8, color, 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
app = Flask(__name__,template_folder='./templates')
webview.create_window('Fighting detection',app)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == "__main__":
    # app.run(debug=True)
    webview.start()