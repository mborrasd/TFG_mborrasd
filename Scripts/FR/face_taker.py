"""

FACIAL RECOGNITION: Face taker

"""

# Import necessary libraries
import cv2
import os
import time

# Check if folder where the images will be stored exists
if not os.path.exists('images'):
    os.makedirs('images')
    
    
# Initialize a faceCascade classifier from haarcascade_frontalface_default.xml model
faceCascade = cv2.CascadeClassifier('/home/pi/project/codes/haarcascade_frontalface_default.xml')
face_detector = faceCascade

# Initialize camera and set input frames' hight and width
cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

# Initialize counter of number of faces detected
count = 0

# Requests user to enter a unique ID integer for the subject (label in the database of known faces)
face_id = input('\n enter user id (MUST be an integer) and press <return> -->  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

# Start main loop of the program
while(True):
    
    # Extract img (frames) and ret (bool indicating if frame was successfully read)
    ret, img = cam.read()
    
    # Transform frames to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frames
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # For X, Y, W and H of frames containing faces
    for (x,y,w,h) in faces:
        
        # Draw a rectangle around the face (to visualize in screen) and increase counter by one unit
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        
        # Save the face within the rectangle in the images directory
        # Images are saved with names "Users." followed by the ID and the count value
        cv2.imwrite("./images/Users." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        
        # Show on screen the face (with the rectangle)
        cv2.imshow('image', img)
        
    # Wait for key press every 100 ms to end the program
    k = cv2.waitKey(100) & 0xff
    if k < 100:
        break
    
    # Run VideoCapture until 100 samples have been collected
    elif count < 100:
        
        # Add a delay of 1 s so that the subject can reposition after each take
        time.sleep(1)
        break
    
# Exit main loop, release camera and destroy windows
print("\n [INFO] Exiting Program.")
cam.release()
cv2.destroyAllWindows()

"""

Bibliography: 
    https://github.com/medsriha/real-time-face-recognition

"""
