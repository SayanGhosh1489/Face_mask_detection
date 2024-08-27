import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import cv2
import os
from src.helper import *

model = load_model(r".\Model\best_model.keras")

camera = cv2.VideoCapture(0)
_, frame = camera.read()
_, frame = camera.read()
filename = "capture.png"
cv2.imwrite(f"static\{filename}", frame)
camera.release()
results = image_preprocessing(filename)
if results is None:
    print("no image")
else:
    img_preds = results[0]
    frame = results[1]
    faces_detected = results[2]

    results2 = predictions_results(img_preds, frame, faces_detected, filename)
    full_filename = os.path.join(r".\Upload", filename)

    number_of_face="Number of faces detected: {}".format(results2[0]),
    no_mask_face="No face mask count: {}".format(results2[1]),
    correct_mask_face="Correct face mask count: {}".format(results2[2]),
    incorrect_mask_face="Incorrect face mask count: {}".format(results2[3])

    print(number_of_face)
    print(no_mask_face)
    print(correct_mask_face)
    print(incorrect_mask_face)
