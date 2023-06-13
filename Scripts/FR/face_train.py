"""

FACIAL RECOGNITION: Face train

"""

# Import necessary libraries
import cv2
import numpy as np
from PIL import Image
import os

# Directory path name where the face images are stored
path = '/home/pi/project/codes/images'

# Create a LBPHF face recognizer using OpenCV
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Initialize a faceCascade classifier from haarcascade_frontalface_default.xml model
detector = cv2.CascadeClassifier('/home/pi/project/codes/haarcascade_frontalface_default.xml');

# Definition of function that finds images in directory
def getImagesAndLabels(path):
    
    # Save image paths in imagePaths list, with f as the index of the file
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    
    # Save found images in faceSamples list and IDs in ids list
    faceSamples=[]
    ids = []
    
    # For every image path, open the image in grayscale and convert it to a  uint8 numpy
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        
        # Split the name of the image path to find ID
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        
        # Detect faces in the image
        faces = detector.detectMultiScale(img_numpy)
        
        # Save each face in faceSamples and their IDs in ids
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
            
            
    return faceSamples,ids

# Initialize training of recognizer
print ("\n[INFO] Training faces...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into the current directory
recognizer.write('/home/pi/project/codes/trainer.yml')
print("\n[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


"""

Bibliography: 
    https://github.com/medsriha/real-time-face-recognition
	
"""
