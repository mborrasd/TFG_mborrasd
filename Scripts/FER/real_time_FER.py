"""

FACIAL EMOTION RECOGNITION

"""

# Import necessary libraries
import cv2
import numpy as np
from keras.utils import img_to_array
from keras.models import load_model
import time

# Load h5 model
model = load_model("/home/pi/project/codes/fer_model.h5")

# Start the webcam feed
cap = cv2.VideoCapture(0)

# Start main loop
while True:
    
    # Extract img (frames) and ret (bool indicating if frame was successfully read)
    ret, img = cap.read()
    if not ret:
        break
    
    # Create haar classifier and transform frames to grayscale
    face_detector = cv2.CascadeClassifier('/home/pi/project/codes/haarcascade_frontalface_default.xml')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces available on camera
    faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=7, minSize=(100, 100), maxSize=(200, 200))

    # Take each face available on the camera and preprocess it
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        roi_gray = gray_img[y:y + w, x:x + h]
        
        # Find ROI containing only face and resize it to match the input of model C
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        # Save pixels of ROI in an array and extend dimensions (X, 48, 48, 1)
        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        
        # Normalize pixel values
        img_pixels /= 255.0
        
        # Predict the emotions
        predictions = model.predict(img_pixels)
        
        # Apply factor to predictions
        factors = [1.7, 2, 2.2, 0.6, 1.2, 1, 0.4]
        predictions *= np.array(factors)
        
        # Find emotion with highest probability
        emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
  

        # Compute probabilities for all emotions
        for i in range(len(emotions)):
            prob_i = predictions[0, i] * 100
            cv2.putText(img, emotions[i] + f'   {prob_i:.2f}%', (int(x)+200, int(y) + 30*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0), 2)
        
        # Conditional statements to improve FER predictions
        if predictions[0, 1] * 100 >= 1 and predictions[0, 0] * 100 <= 20:
            predicted_emotion = emotions[1]
            prob = predictions[0, 1] * 100
            
        elif predictions[0, 0] * 100 >= 15:
            predicted_emotion = emotions[0]
            prob = predictions[0, 0] * 100        
        
        elif predictions[0, 2] * 100 >= 5:
            predicted_emotion = emotions[2]
            prob = predictions[0, 2] * 100
            
        elif predictions[0, 4] * 100 >= 5:
            predicted_emotion = emotions[4]
            prob = predictions[0, 4] * 100         
        
        elif predictions[0, 3] * 100 >= 15:
            predicted_emotion = emotions[3]
            prob = predictions[0, 3] * 100
        else: 
            max_index= int(np.argmax(predictions))
            prob = predictions[0, max_index] * 100        
            predicted_emotion = emotions[max_index]
        
        # Show results on screen
        cv2.putText(img, predicted_emotion + f'   {prob:.2f}%', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        resized_img = cv2.resize(img, (1000, 700))
        cv2.imshow('Facial Emotion Recognition', resized_img)
        
        # Delay for 0.5 seconds before next prediction
        time.sleep(0.5)

    # Press 'Q' to exit program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
