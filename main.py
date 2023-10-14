import os
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
from PIL import UnidentifiedImageError
from flask import Flask, render_template, Response, request
import cv2
from keras.models import model_from_json
import numpy as np
from keras.src.utils import img_to_array

app = Flask(__name__, template_folder='C:\\Users\\harsh\\PycharmProjects\\pythonProject\\template', static_url_path='/static')

video_path = ""
image_path = ""

json_file = open("C:\\Users\\harsh\\Downloads\\facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("C:\\Users\\harsh\\Downloads\\facialemotionmodel.h5")


haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def detect_emotion(frame):
    global face_cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

    for (x, y, w, h) in faces:
        image = gray[y:y+h, x:x+w]
        image = cv2.resize(image, (48, 48))
        img = extract_features(image)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]
        cv2.putText(frame, prediction_label, (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

    return frame
def generate_frames1():
    global video_path
    cap = cv2.VideoCapture(video_path)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = detect_emotion(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
def generate_frames():
    webcam = cv2.VideoCapture(0)
    while True:
        success, frame = webcam.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(frame, 1.3, 5)
            try:
                for (p, q, r, s) in faces:
                    image = gray[q:q+s, p:p+r]
                    cv2.rectangle(frame, (p, q), (p+r, q+s), (255, 0, 0), 2)
                    image = cv2.resize(image, (48, 48))
                    img = extract_features(image)
                    pred = model.predict(img)
                    prediction_label = labels[pred.argmax()]
                    cv2.putText(frame, '% s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            except cv2.error:
                pass
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def home():
    return render_template("main.html")

@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/page1')
def page1():
    return render_template('page1.html')

@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/page3')
def page3():
    return render_template('page3.html')

@app.route('/page4')
def page4():
    return render_template('page4.html')



@app.route('/upload', methods=['POST'])
def upload():
    global video_path
    if 'video' not in request.files:
        return "No file part"

    video = request.files['video']

    if video.filename == '':
        return "No selected file"

    if video:
        video_path = os.path.join("static", "uploaded_video.mp4")
        video.save(video_path)

    return Response(generate_frames1(), mimetype='multipart/x-mixed-replace; boundary=frame')


def detect_emotion1(frame):
    global face_cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

    for (x, y, w, h) in faces:
        image = gray[y:y + h, x:x + w]
        image = cv2.resize(image, (48, 48))
        img = extract_features(image)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]


        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


        cv2.putText(frame, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255) )

    return frame
def generate_frames2():
    global image_path
    cap = cv2.imread(image_path)
    frame = detect_emotion1(cap)

    ret, buffer = cv2.imencode('.jpg', frame)
    if ret:
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.destroyAllWindows()
@app.route('/upload1', methods=['POST'])

def upload1():
    global image_path
    if 'image' not in request.files:
        return "No file part"

    image = request.files['image']

    if image.filename == '':
        return "No selected file"

    if image:
        image_path = os.path.join("static", "uploaded_image.jpeg")
        image.save(image_path)

    return Response(generate_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
