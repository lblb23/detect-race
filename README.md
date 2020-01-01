# detect-race
Draft project for detecting a person's race with simple convolutional neural network in Keras

## Getting Started

1. Clone repo.
2. Unzip data_faces.zip.
3. Run model on any image:
```
python predict.py --img_path jackie_chan.jpg
```
4. You can evaluate model on dataset:
```
python eval.py
```
5. You can train model on dataset:
```
python train.py
```

## Dataset

Dataset contains 12 513 cropped images of faces. Training set - 10 060 images, test set - 2 513 images. Classes are distributed almost uniformly.

My implementation reached 66% accuracy.

## Face detector

I used the face detector from OpenCV (haarcascade_frontalface_default.xml).
