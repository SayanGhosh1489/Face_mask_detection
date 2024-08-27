import tensorflow as tf
import os
from tensorflow import keras
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input

model = load_model(r".\Model\best_model.keras")

# Load the face classifier
face_classifier = cv2.CascadeClassifier(r"D:\Projects\Face_mask_detection\templates\haarcascade_frontalface_default.xml")

# Labels for different outcomes
text_mask = "Mask On"
text_no_mask = "Mask Off"
text_wrong_mask = "Incorrect Mask"

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 0.8

def fileNameGenerator(path, filename):
    # Convert all filenames in the directory to lowercase
    fileList = [file.lower() for file in os.listdir(path)]
    # print(fileList)
    
    # Initialize the base name and extension
    base_name = filename.split('_')[0]
    ext = filename.split('.')[1]
    
    # Check if the filename already exists in the fileList
    if filename.lower() in fileList:
        # Extract and increment the digit part
        digit = int(filename.split('_')[1].split('.')[0])
        # print(f"Initial digit: {digit}")
        digit += 1
        # print(f"Incremented digit: {digit}")
    else:
        # If filename does not exist, start with _0
        digit = 0
        # print(f"Starting digit: {digit}")

    # Generate a new filename with the incremented digit
    new_filename = f"{base_name}_{digit}.{ext}"
    
    # If the new filename still exists in the fileList, keep incrementing
    while new_filename.lower() in fileList:
        digit += 1
        new_filename = f"{base_name}_{digit}.{ext}"
        # print(f"New filename being checked: {new_filename}")

    return new_filename



def predict(image):
    face_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_frame = cv2.resize(face_frame, (224, 224))
    face_frame = img_to_array(face_frame)
    face_frame = np.expand_dims(face_frame, axis=0)
    face_frame = preprocess_input(face_frame)
    prediction = model.predict(face_frame)
    boost_factor = 1.1  # Define a boost factor for incorrect mask
    prediction[0][1] *= boost_factor  # 'mask_weared_incorrect' is the second class
    prediction[0] /= np.sum(prediction[0])  # Normalize to maintain valid probabilities
    
    predicted_class = np.argmax(prediction)
    return predicted_class, prediction[0]

# Function to detect faces and predict mask status
def detector(gray_image, frame):
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5)
    
    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        mask, probabilities = predict(roi_color)
        # Prediction classes
        classes = ['without_mask', 'mask_weared_incorrect', 'with_mask']
        result = classes[mask]
        probability = round(probabilities[mask] * 100, 0)

        if result == "without_mask":
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{text_no_mask}: {probability:.0f}%", org=(x+5, y-10), fontFace=font, fontScale=scale, color=(0, 0, 255), thickness=2)
        
        elif result == "mask_weared_incorrect":
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            cv2.putText(frame, f"{text_wrong_mask}: {probability:.0f}%", org=(x+5, y-10), fontFace=font, fontScale=scale, color=(0, 0, 255), thickness=2)
        
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{text_mask}: {probability:.0f}%", org=(x+5, y-10), fontFace=font, fontScale=scale, color=(0, 255, 0), thickness=2)
            
    return frame

# Capture an image from the webcam
camera = cv2.VideoCapture(0)
ret, frame = camera.read()
if ret:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect = detector(gray_frame, frame)
    
    # Save the processed frame to a file
    filename = "IM_0.png"
    filename = fileNameGenerator(path=r".\Upload", filename=filename)
    cv2.imwrite(f"Upload\{filename}", frame)
camera.release()

# Read the saved image and convert it to an array
img_arr = cv2.imread(f"Upload\{filename}")
img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB) 

# Display the image using matplotlib
plt.imshow(img_arr)
plt.axis('off')
plt.show()
