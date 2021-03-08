import imghdr
import os
import numpy as np
import cv2
from PIL import Image
from os.path import join, dirname, realpath
from flask import Flask, render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename
import json

UPLOADS_PATH = join(dirname(realpath(__file__)), 'static/uploads/..')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'

def validate_image(stream):
    header = stream.read(512)
    stream.seek(0) 
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            abort(400)
        uploaded_file.save('new.png')
   
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier('/Users/matt/Python/LazerEyes/haarcascade_eye.xml')

    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    eye_cascade = cv2.CascadeClassifier('/Users/matt/Python/LazerEyes/haarcascade_eye.xml')

    
    img = cv2.imread('new.png')
    dot = cv2.imread('dot_transparent.png', cv2.IMREAD_UNCHANGED)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        eyes = eye_cascade.detectMultiScale(gray[y:y+h, x:x+w])

        for (ex, ey, ew, eh) in eyes:

            # Filter out small detections, if you're only want to have the eyes
            if ew < 100 or eh < 100:
                continue

            d = cv2.resize(dot.copy(), (eh, ew))
            d_alpha = d[..., 3] / 255.0
            d_alpha = np.repeat(d_alpha[..., np.newaxis], 3, axis=2)
            d = d[..., :3]

            img[y+ey:y+ey+eh, x+ex:x+ex+ew, :] = \
                img[y+ey:y+ey+eh, x+ex:x+ex+ew, :] * (1 - d_alpha) + d * d_alpha

    cv2.imwrite('out.png', img)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)