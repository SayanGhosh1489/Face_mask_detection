import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.helper import *


def capturingImage():
    ret, frame = camera.read()
    if ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detect = detector(gray_frame, frame)
    
    # Save the processed frame to a file
    filename = "IM_CAP_0.png"
    path = "Capture"
    filename = fileNameGenerator(path=path, filename=filename)
    filepath = os.path.join(path,filename)
    cv2.imwrite(filepath, frame)
    camera.release()
    return filepath

def imgShow(filename):
    # Read the saved image and convert it to an array
    img_arr = cv2.imread(filename)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB) 

    # Display the image using matplotlib
    plt.imshow(img_arr)
    plt.axis('off')
    plt.show()

# filename = capturingImage()
# imgShow(filename)