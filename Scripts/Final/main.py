"""

Main Program

"""

print('Initializing emotion recognition project...')


# import libraries
print('   - Importing libraries...')
import datetime, time
import gspread
import cv2
import numpy as np
import librosa
import noisereduce as nr
import wave
import pyaudio
from keras.models import load_model
from keras.utils import img_to_array
from oauth2client.service_account import ServiceAccountCredentials
from pydub import AudioSegment, effects
from array import array
from threading import Thread
import sounddevice # to avoid rising ALSA lib errors

# class to define threads with output values
class ThreadWithReturnValue(Thread):

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
                            
# import FER and SER modules
print('   - Importing modules...')
from fer_prediction_module import fer_prediction_function
from ser_prediction_module import ser_prediction_function
 
# establish connection with database
print('   - Connecting to database...')
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('/home/pi/project/codes/project_to_database.json', scope) 
client = gspread.authorize(creds)

# load patient profiles in database
sheet_patient_A = client.open("project_database").sheet1
sheet_patient_B = client.open("project_database").get_worksheet(1)
sheet_patient_C = client.open("project_database").get_worksheet(2)
sheet_patient_D = client.open("project_database").get_worksheet(3)
sheet_patient_E = client.open("project_database").get_worksheet(4)
sheet_patient_F = client.open("project_database").get_worksheet(5)
sheet_patient_G = client.open("project_database").get_worksheet(6)
sheet_patient_H = client.open("project_database").get_worksheet(7)
sheets = [sheet_patient_A, sheet_patient_B, sheet_patient_C, sheet_patient_D, sheet_patient_E, sheet_patient_F, sheet_patient_G, sheet_patient_H]

# load face recognition model
print('   - Loading prediction models...')
fr_model = cv2.face.LBPHFaceRecognizer_create()
fr_model.read('/home/pi/project/codes/trainer.yml')

# load FER model
fer_model = load_model("/home/pi/project/codes/fer_model.h5")

# load SER model
ser_model = load_model("/home/pi/project/codes/ser_model.h5")

# load face cascade classifier
face_cascade_Path = "/home/pi/project/codes/haarcascade_frontalface_default.xml"

# load patient ids
names = ['Unknown', 'Patient_A', 'Patient_B', 'Patient_C', 'Patient_D', 'Patient_E', 'Patient_F', 'Patient_G', 'Patient_H']

print('------------------------------------------------------------')
print('Ready to make predictions')

# selection of prediction mode
user_input = input("   - Please press 1 for monomodal prediction or 2 for multimodal prediction:")
if user_input == '1':
    mode = 1
    
else:
    mode = 0
    
# initialize main loop
collect_data = False
while True:
    
    if mode == 1:
        
        # start data collection with monomodal prediction 
        user_input = input("   - Please press the 'S' key to start collecting data, or press 'Q' to quit the program: ")
        if user_input.upper() == 'S':
            collect_data = True
            print("   - Data collection is now running. Press 'C' + 'Ctrl' keys to stop:") 
            
            # start the webcam feed
            cap = cv2.VideoCapture(0)
            cap.set(3, 640)
            cap.set(4, 480)
    
            # min height and width for the  window size to be recognized as a face
            minW = 0.1 * cap.get(3)
            minH = 0.1 * cap.get(4)
            
            # data collection loop
            try:
                while collect_data:
                    
                    ret, img = cap.read()
                    if not ret:
                        break
                               
                    # preprocess images and create haar classifier
                    face_detector = cv2.CascadeClassifier(face_cascade_Path)
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                    # detect faces available on camera
                    faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))
                
                    # take each face available on the camera and preprocess it
                    for (x, y, w, h) in faces:
                        
                        # face recognition
                        id_num, confidence = fr_model.predict(gray_img[y:y + h, x:x + w])
                        if (confidence < 100):
                            patient_id = names[id_num]
        
                        else:
                            # unknown face
                            break
                        
                        # FER prediction
                        fer_data = fer_prediction_function(x, y, w, h, gray_img, fer_model)
                            
                        # delay for 0.5 secondS
                        time.sleep(0.5)
                            
                        # send data to database                
                        time_data = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                        data =[time_data, patient_id, fer_data, '-']
                        sheet = sheets[id_num-1]
                        sheet.append_row(data)
                        
            # stop data collection
            except KeyboardInterrupt:
                cap.release()
                print("Data collection stopped.")
                print('------------------------------------------------------------')
                collect_data = False
                pass
            
        # press Q again to kill the program
        elif user_input.upper() == 'Q':
            # stream.stop_stream()
            # stream.close()
            # p.terminate()
            print('The program has been stopped.')
            break            
                        

    else:
        
        # start data collection with multimodal prediction
        user_input = input("   - Please press the 'S' key to start collecting data, or press 'Q' to quit the program: ")
        if user_input.upper() == 'S':
            collect_data = True
            print("   - Data collection is now running. Press 'C' + 'Ctrl' keys to stop:") 
            
            # start the webcam feed
            cap = cv2.VideoCapture(0)
            cap.set(3, 640)
            cap.set(4, 480)
    
            # min height and width for the  window size to be recognized as a face
            minW = 0.1 * cap.get(3)
            minH = 0.1 * cap.get(4)
            
            # initialize audio recording variables
            RATE = 48000 # default samples/s
            CHUNK = 512 # hop length
            RECORD_SECONDS = 3.61
            FORMAT = pyaudio.paInt32
            CHANNELS = 1
            WAVE_OUTPUT_FILE = "/home/pi/project/codes/output.wav"

            # open an input audio channel
            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK,
                            input_device_index=1)
            
            # data collection loop
            try:
                while collect_data:
                    
                    ret, img = cap.read()
                    if not ret:
                        break
                               
                    # preprocess images and create haar classifier
                    face_detector = cv2.CascadeClassifier(face_cascade_Path)
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # detect faces available on camera
                    faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))
                    
                    # take the first face availablae on the camera and preprocess it
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        
                        # face recognition
                        id_num, confidence = fr_model.predict(gray_img[y:y + h, x:x + w])
                        if (confidence < 100):
                            patient_id = names[id_num]
        
                        else:
                            # unknown face
                            break

                        # FER module
                        def module1():
                            
                            reps = 0
                            fer_predictions = []
                            time_data = []
                                                                   
                            # repeat FER loop 7 times
                            while reps < 3: 
                                
                                # FER prediction
                                fer_data = fer_prediction_function(x, y, w, h, gray_img, fer_model)
                                fer_predictions.append(fer_data)
                                
                                # save time
                                time_rep = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                                time_data.append(time_rep)
                                
                                # delay for 0.3 second and increase counter
                                time.sleep(0.3)
                                reps += 1
                            
                            return fer_predictions, time_data
                                    
                        # SER module
                        def module2():
       
                            # SER prediction
                            ser_data = ser_prediction_function(RATE, CHUNK, RECORD_SECONDS, FORMAT, 
                                                        CHANNELS, WAVE_OUTPUT_FILE, p, stream, ser_model)
                            
                            return ser_data
                            
                        # create two threads for the two modules
                        thread1 = ThreadWithReturnValue(target = module1)
                        thread2 = ThreadWithReturnValue(target = module2)
                        
                        # start the threads
                        thread1.start()
                        thread2.start()
                        
                        # wait for both threads to finish and get outputs
                        fer_predictions, time_data = thread1.join()
                        ser_data = thread2.join()
                                        
                        # send data to database
                        for i in range(0,3):
                            data = [time_data[i], patient_id, fer_predictions[i], ser_data]
                            sheet = sheets[id_num-1]
                            sheet.append_row(data)
                 
            # stop data collection
            except KeyboardInterrupt:
                cap.release()
                stream.stop_stream()
                stream.close()
                p.terminate()
                collect_data = False
                print("Data collection stopped.")
                print('------------------------------------------------------------')                
                pass
                    
        # press Q again to kill the program
        elif user_input.upper() == 'Q':
            print('The program has been stopped.')
            break

    
    
