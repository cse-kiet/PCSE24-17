from flask import Flask,render_template,request,Response
import cv2
from tensorflow.keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2 
import numpy as np
import speech_recognition as sr
import sys
app=Flask(__name__)
camera=cv2.VideoCapture(0)
def gen_frame():
    while True:
        success,frame=camera.read()#read the camera frame
        if not success:
            break
        else:
            roi_gray = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                label_position = (0,0)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n\r\n')
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def video_feed():
    return Response(gen_frame(),mimetype='multipart/x-mixed-replace;')

if __name__ == '__main__':
    app.run(port=3000,debug=True)