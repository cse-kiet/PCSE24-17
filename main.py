import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import speech_recognition as sr
import sys
import time 
import cv2
from tensorflow.keras.models import load_model

# Page Titles
page_titles = ["Home","Speech Analysis", "About", "Project Description", "GitHub Link", "Research Paper Link"]

# Function for each sidebar page
def home():
    st.title("Home Page")
    st.write("Welcome to your Streamlit app!")

def video_analysis():
    st.header("Video Analysis Screen")
    face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
    classifier =load_model(r"model.h5")
    video_capture = cv2.VideoCapture(0)  

    while True:
        ret, frame = video_capture.read()
        emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
        if ret:
            if recording:
                # Process frame using your model
                prediction = classifier.predict(frame)[0]
                label=emotion_labels[prediction.argmax()]
                # Display the video feed with predictions
                st.image(frame, channels="BGR")
                st.write(f"Model Prediction: {prediction}")

            # Control recording with button
            if st.button("Start Recording"):
                recording = True
            elif st.button("Stop Recording"):
                recording = False

        else:
            break

        # Exit on 'q' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def speech():
    # initialize the recognizer
    r = sr.Recognizer()
    st.write("Please talk and say QUIT to stop: ")
    l=0
    i=0
    time_1=time.time()
    time_2=0
    while i==0:
        with sr.Microphone() as source:
        #to clear background noise
            r.adjust_for_ambient_noise(source,duration=0.3)
        #read the audio data from the default microphone
            audio = r.listen(source)
            st.write("Recognizing...")

        #convert speech to text
            try:
                text = r.recognize_google(audio)
                st.write("Speaker: ", text)
                l=l+len(text.split(' '))
                if 'quit' in text:
                    time_2=time.time()
                    i=1
            except:
            #if any error occurs
                st.write('Please say it again!!')
    if((time_2-time_1)/l>0.15): print("Fast Speech")
    elif((time_1-time_2)/l>0.15): print("Slow Speech")
    else: print("Nice Speed!")
    
    st.write('hello')
    
def about():
    st.title("About")
    st.write("This is a brief description about the project, its purpose, and the team behind it.")

def project_description():
    st.title("Project Description")
    st.write("The project aims at delivering a web application which can provide candidates (users) with two main functionalities: Rehearsing online interviews with camera monitoring and speech based feedback. These two functionalities are provided on a web application to reduce the time required to access the services and make the user experience better. Web Application has been developed with streamlit v1.35.0. to ease the integration of machine learning and deep learning services in the platform.""")
    
    st.subheader("Video Recording")
    st.write("The first functionality is implemented by using streamlit’s components for video recording which receives inputs from camera (integrated camera or additional cam) and provides video and audio feed to be used as inputs for processing functionalities. This video feed, which comes up as images are processed to find the visual features like facial emotion and eye focus of the user. Facial Emotion Recognition is implemented by using deep learning model based on FER2013 dataset, which was further tuned with interview recording of college students to make it ready for use for the sample audience in the development audience. The other service utilized to provide this functionality is based on eye tracking techniques. Eye tracking techniques cover a wide range of algorithms ranging from tracking of eye to tracking iris only to find the angle at which the user is seeing before the screen. Eye tracking for this use case doesn’t necessarily involve very precise angles but simple check over the candidate if he/she is focusing on the screen or involving in unfair means/showing under confident reflexes before the interviewer. These two main models are currently used in the project, which also opens scope for additional features but being restricted due to near real time processing and feedback requirements. This feedback is stored and marked for each interview for each user.")
    
    st.subheader("Audio Functionality")
    st.write("The second functionality focuses on the speech delivered. Candidate’s speech delivery affects the overall interview experience due to factors such as speed of speaking, stammering or the repetitive pronunciation of same lines. This module takes in audio input and converts it into text for further processing. Pretrained model’s API is used for this purpose to understand a wide range of pronunciation and speech qualities. The count check functionality checks the words spoken per minute and provides feedback on the number of words spoken and the 3 preferred word count. This module has other features which check the occurrences of stammered words in between sentences. This is reflected to the user to avoid pronouncing the word or sound multiple times in an interview. Thus, this project will be able to serve the purpose of skill training and interview preparation.")
    

def github_link():
    st.title("GitHub Repository")
    
    st.write('GitHub is a web-based hosting platform for software development projects. It uses a version control system called Git, allowing developers to track changes in code, collaborate on projects, and revert to previous versions if needed. GitHub offers features like issue tracking, pull requests for code review, and wikis for project documentation, making it a central hub for  software development workflows.')
    st.markdown("[Project GitHub Link](https://github.com/cse-kiet/PCSE24-17")  

def research_paper_link():
    st.title("Research Paper")
    pdf_viewer(r'Final-Comparative Analysis of Deep Learning Models for Facial Emotional Recognition.docx.pdf')
    if st.button("Link to Research Paper"): 
        st.markdown("[Research Paper Link](https://ieeexplore.ieee.org/document/10522163)")  # Replace with your link (if applicable)

# Sidebar navigation
selected_page = st.sidebar.selectbox("Navigation", page_titles)

# Display content based on selected page
if selected_page == "Home":
    home()
elif selected_page == 'Speech Analysis':
    speech()
    
elif selected_page == "About":
    about()
elif selected_page == "Project Description":
    project_description()
elif selected_page == "GitHub Link":
    github_link()
elif selected_page == "Research Paper Link":
    research_paper_link()

# Display Streamlit documentation link in the footer
st.sidebar.markdown(
    "---", unsafe_allow_html=True
)  # Adds a horizontal separator in the sidebar
st.sidebar.markdown(
    "Please contact kushgrathisside for additional information"
)
