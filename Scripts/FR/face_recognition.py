"""
	
FACIAL RECOGNITION: Face recognition
	
"""

# Import necessary libraries
import cv2

# Create a LBPHF face recognizer using OpenCV
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/home/pi/project/codes/trainer.yml')

# Initialize a faceCascade classifier from haarcascade_frontalface_default.xml model
face_cascade_Path = '/home/pi/project/codes/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(face_cascade_Path)

# Set the font with which text will be displayed
font = cv2.FONT_HERSHEY_SIMPLEX

# List patient IDs according to integer ID in database
id = 0
names = ['Unknown', 'Patient_A', 'Patient_B', 'Patient_C', 'Patient_D', 'Patient_E', 'Patient_F', 'Patient_G', 'Patient_H']

# Initialize camera and set input frames' hight and width
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# Min Height and Width for the  window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# Start main loop of the program
while True:
    
    # Extract img (frames) and ret (bool indicating if frame was successfully read)
    ret, img = cam.read()
    
    # Transform frames to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frames
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    # For X, Y, W and H of frames containing faces
    for (x, y, w, h) in faces:
        
        # Draw a rectangle around the face (to visualize in screen) and increase counter by one unit
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Predict the probability of each detected face being a known face in database
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        
        # If confidence level is below 100, assign face to its ID in database
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        
        # Unknown face
        else:
            id = "Who are you ?"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    # Visualize frames and IDs on screen
    cv2.imshow('camera', img)
    
    # Escape to exit the webcam / program
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
    
print("\n [INFO] Exiting Program.")
cam.release()
cv2.destroyAllWindows()

"""

Bibliography: 
    https://github.com/medsriha/real-time-face-recognition
	
"""