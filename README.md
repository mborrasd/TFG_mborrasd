# Design, Development, and Evaluation of a Real-Time Facial Expression and Speech Emotion Recognition System
This repository contains all the notebooks, scripts and other relevant files of the thesis "Design, Development, and Evaluation of a Real-Time Facial Expression and Speech Emotion Recognition System".

## Contents of the repository
Folder “Notebooks” contains the following files:
  -	convert_fer2013_to_images_and_landmarks.ipynb
  -	dataset_exploration.ipynb
  -	prepare_fer2013+.ipynb
  -	train_cnn_c.ipynb
  -	train_cnn_fer2013+.ipynb
  -	train_cnn_fer2013.ipynb
  -	train_ser.ipynb
  -	train_svm_fer2013+.ipynb
  -	train_svm_fer2013.ipynb

Folder “Scripts” contains the following folders and files:
  *	FER
    -	fer_model.h5
    -	haarcascade_frontalface_default.xml
    -	real_time_FER.py
  *	FR
    -	face_recognition.py
    -	face_taker.py
    -	face_train.py	
  *	Final
    -	fer_model.h5
    -	fer_prediction_module.py
    -	haarcascade_frontalface_default.xml
    -	main.py
    -	output.wav
    -	python_to_database.py
    -	ser_model.h5
    -	ser_prediction_module.py
  *	SER
    -	output.wav
    -	real_time_ser.py
    - ser_model.h5
  * project_database.xlsx
  * requirements.txt

Note that project_to_database.py has not been uploaded to the repository due to security concerns.

# Abstract
This thesis presents the design, development, and evaluation of a real-time emotion recognition system for healthcare applications. It aims to remotely monitor patients' emotional states using Facial Expression Recognition (FER) and Speech Emotion Recognition (SER). The collected data is stored in a cloud-based database, allowing healthcare professionals to access real-time updates from anywhere. Additionally, the system integrates Facial Recognition (FR) to identify the patients, before emotion recognition, to enable data storage into separate profiles.
The platform has two types of functioning modes: monomodal and multimodal emotion recognition. In the monomodal approach, FER is employed to infer the emotional state of the subject. On the other hand, the multimodal approach combines both FER and SER to provide deeper insights into the subject's emotional state.
The system is designed as a probe of concept with a general purpose in mind. However, this work also proposes various applications in which the system could be integrated, and outlines the necessary adjustments required to meet the specific requirements of each use case. 
Furthermore, ethical implications and the protection of personal data are addressed within this thesis.

*Key words*: Facial Recognition, Facial Expression Recognition, Speech Emotion Recognition, Computer Vision, Edge Computing.
