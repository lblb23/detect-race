# Predict class for one image

import argparse

import cv2
import numpy as np
from keras.models import load_model

model = load_model('models/weights.hdf5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

parser = argparse.ArgumentParser()
parser.add_argument('--img_path',
                    default='jackie_chan.jpg',
                    dest='img_path',
                    help='path to image')

args = parser.parse_args()
filepath = args.img_path

img = cv2.imread(filepath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    face_img = img[y:y + h, x:x + w]

face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
face_img = cv2.resize(face_img, (224, 224))
face_img = face_img / 255.0
face_img = np.reshape(face_img, [1, 224, 224, 3])

classes = model.predict(face_img)[0]

result = {}
result['asian'] = classes[0]
result['black'] = classes[1]
result['euro'] = classes[2]

print(result)
