import speech_recognition as sr
import sys
import time 

# initialize the recognizer
r = sr.Recognizer()
print("Please talk and say QUIT to stop: ")
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
        print("Recognizing...")

        #convert speech to text
        try:

            text = r.recognize_google(audio)
            print("Speaker: ", text)
            l=l+len(text.split(' '))
            if 'quit' in text:
                time_2=time.time()
                i=1
        except:
            #if any error occurs
            print('Please say it again!!')
if((time_2-time_1)/l>0.15): print("Fast Speech")
elif((time_1-time_2)/l>0.15): print("Slow Speech")
else: print("Nice Speed!")







 