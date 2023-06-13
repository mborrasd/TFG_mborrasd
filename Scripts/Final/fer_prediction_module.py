"""

FER prediction module

"""

# import libraries
import cv2
import numpy as np
from keras.utils import img_to_array

def fer_prediction_function(x, y, w, h, gray_img, fer_model):

    # find roi
    roi_gray = gray_img[y:y + w, x:x + h]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    img_pixels = img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255.0
    
    # predict the emotions
    fer_predictions = fer_model.predict_on_batch(img_pixels)
    
    # apply factor to predictions
    factors = [1.7, 2, 2.2, 0.6, 1.2, 1, 0.4]
    fer_predictions *= np.array(factors)
    
    # fer with prediction priorities
    if fer_predictions[0, 1] * 100 >= 1 and fer_predictions[0, 0] * 100 <= 20:
        fer_data = 1
        
    elif fer_predictions[0, 0] * 100 >= 15:
        fer_data = 0 
    
    elif fer_predictions[0, 2] * 100 >= 5:
        fer_data = 2
        
    elif fer_predictions[0, 4] * 100 >= 5:
        fer_data = 4      
    
    elif fer_predictions[0, 3] * 100 >= 15:
        fer_data = 3
        
    else: 
        max_index = int(np.argmax(fer_predictions))     
        fer_data = max_index
        
    return fer_data
