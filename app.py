import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os

# Load the model and face detector
model_path = r'.\Model\mask_detector.h5'
cascade_path = r'.\template\haarcascade_frontalface_default.xml'
video_path = "sgg.mp4"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

if not os.path.exists(cascade_path):
    raise FileNotFoundError(f"Cascade file not found: {cascade_path}")

if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

model = load_model(model_path)
face_cascade = cv2.CascadeClassifier(cascade_path)

# Open the video file
image = cv2.VideoCapture(video_path)

if not image.isOpened():
    raise ValueError("Error opening video stream or file")

while True:
    ret, frame = image.read()
    if not ret:
        break

    # Resize frame for processing
    frame = cv2.resize(frame, (500, 500), interpolation=cv2.INTER_AREA)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, 1.2, 7, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)

    IDs = []

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        cropped_face = face.copy()
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face = preprocess_input(face)

        pred = model.predict(face)[0]
        WithoutMask, CorrectMask, InCorrectMask = pred

        # Determine the label and color for the bounding box and text
        if max(pred) == CorrectMask:
            label = "Correct Mask"
            color = (0, 255, 0)
            IDs.append(1)
        elif max(pred) == InCorrectMask:
            label = "Incorrect Mask"
            color = (0, 0, 255)
            IDs.append(2)
        else:
            label = "No Mask"
            color = (0, 0, 255)
            IDs.append(0)

        # Display the label with confidence score
        label_text = "{}: {:.2f}%".format(label, max(WithoutMask, CorrectMask, InCorrectMask) * 100)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    # Display counts on the frame
    correct_mask_count = IDs.count(1)
    no_mask_count = IDs.count(0)
    incorrect_mask_count = IDs.count(2)
    face_count = correct_mask_count + no_mask_count + incorrect_mask_count
    text = f"Faces: {face_count} | No Mask: {no_mask_count} | Correct: {correct_mask_count} | Incorrect: {incorrect_mask_count}"
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

image.release()
cv2.destroyAllWindows()
