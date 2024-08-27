# Capture video from webcam
import cv2
from src.helper import *

def videoCapture():

    while True:
        ret, frame = camera.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detect = detector(gray_frame, frame)
        
        cv2.imshow("Video", detect)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
    camera.release()
    cv2.destroyAllWindows()

# videoCapture()
