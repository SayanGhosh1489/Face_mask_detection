import os
from urllib.request import Request
from flask import Flask, render_template, Response, request, redirect, flash
import cv2
from src.helper import *


app = Flask(__name__)
upload_folder = r'.\static\Upload'
app.config['UPLOAD_FOLDER'] = upload_folder

@app.route('/')
def index():
    """my main page"""
    return render_template('index.html')

@app.route('/ImageStream')
def ImageStream():
    """the live page"""
    return render_template('RealtimeImage.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)




