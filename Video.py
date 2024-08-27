import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input

# Load the face classifier
face_classifier = cv2.CascadeClassifier(r"D:\Projects\Face_mask_detection\templates\haarcascade_frontalface_default.xml")

# Load the mask detection model
mask_detection = tf.keras.models.load_model(r".\Model\best_model.keras")

# Labels for different outcomes
text_mask = "Mask On"
text_no_mask = "Mask Off"
text_wrong_mask = "Incorrect Mask"

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 0.8

# Function to predict mask status and return probabilities
def predict(image):
    face_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_frame = cv2.resize(face_frame, (224, 224))
    face_frame = img_to_array(face_frame)
    face_frame = np.expand_dims(face_frame, axis=0)
    face_frame = preprocess_input(face_frame)
    prediction = mask_detection.predict(face_frame)
    boost_factor = 1.1  # Define a boost factor for incorrect mask
    prediction[0][1] *= boost_factor  # 'mask_weared_incorrect' is the second class
    prediction[0] /= np.sum(prediction[0])  # Normalize to maintain valid probabilities
    
    predicted_class = np.argmax(prediction)
    return predicted_class, prediction[0]
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
        probability = round(probabilities[mask] * 100,0)

        if result == "without_mask":
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{text_no_mask}: {probability:.2f}%", org=(x+5, y-10), fontFace=font, fontScale=scale, color=(0, 0, 255), thickness=2)
        
        elif result == "mask_weared_incorrect":
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            cv2.putText(frame, f"{text_wrong_mask}: {probability:.2f}%", org=(x+5, y-10), fontFace=font, fontScale=scale, color=(0, 0, 255), thickness=2)
        
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{text_mask}: {probability:.2f}%", org=(x+5, y-10), fontFace=font, fontScale=scale, color=(0, 255, 0), thickness=2)
            
    return frame

# Capture video from webcam
video_cap = cv2.VideoCapture(0)

while True:
    ret, frame = video_cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect = detector(gray_frame, frame)
    
    cv2.imshow("Video", detect)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
video_cap.release()
cv2.destroyAllWindows()
