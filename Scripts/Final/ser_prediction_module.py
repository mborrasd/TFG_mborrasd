"""

SER prediction module

"""

# import libraries
import numpy as np
import librosa
import noisereduce as nr
import wave
from pydub import AudioSegment, effects
from array import array

# definition of function to perform preprocess of audio files to 3D arrays.
def preprocess(file_path, frame_length = 2048, hop_length = 512):

    # fetch sample rate.
    _, sr = librosa.load(path = file_path, sr = None)
    # load audio file
    rawsound = AudioSegment.from_file(file_path, duration = None) 
    # normalize to 5 dBFS 
    normalizedsound = effects.normalize(rawsound, headroom = 5.0) 
    # transform the audio file to np.array of samples
    normal_x = np.array(normalizedsound.get_array_of_samples(), dtype = 'float32') 
    # noise reduction                  
    final_x = nr.reduce_noise(normal_x, sr=sr)
        
    # features extraction    
    f1 = librosa.feature.rms(y=final_x, frame_length=frame_length, hop_length=hop_length, center=True, pad_mode='reflect').T # Energy - Root Mean Square
    f2 = librosa.feature.zero_crossing_rate(y=final_x, frame_length=frame_length, hop_length=hop_length, center=True).T # ZCR
    f3 = librosa.feature.mfcc(y=final_x, sr=sr, S=None, n_mfcc=13, hop_length = hop_length).T # MFCC   
    X = np.concatenate((f1, f2, f3), axis = 1)
    
    X_3D = np.expand_dims(X, axis=0)
    
    return X_3D, f1

# returns 'True' if below the 'silent' threshold
def is_silent(data_rms):
    
    # compute average RMS value
    avg_rms = np.mean(data_rms)
    
    # check if the average RMS value is below the 'silent' threshold
    return avg_rms < 18000000

# SER prediction in real time
def ser_prediction_function(RATE, CHUNK, RECORD_SECONDS, FORMAT, 
                            CHANNELS, WAVE_OUTPUT_FILE, p, stream, ser_model):

    # start recording
    frames = []
    
    # reset 'data' variable.
    data = np.nan
    
    # compute timesteps
    timesteps = int(RATE / CHUNK * RECORD_SECONDS) # => 339 sequential values

    # insert frames to 'output.wav'.
    for i in range(0, timesteps):
        data = array('l', stream.read(CHUNK, exception_on_overflow = False)) # eliminate overflow error
        frames.append(data)

        wf = wave.open(WAVE_OUTPUT_FILE, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    # 'output.wav' file preprocessing.
    x, f1 = preprocess(WAVE_OUTPUT_FILE)
    
    # check if the input contains silence or not
    if is_silent(f1):
        ser_prediction = "silence"
        
    else:
        # model's prediction => 7 emotion probabilities array.
        predictions = ser_model.predict_on_batch(x)
        
        # present emotion distribution for a sequence (3.61 secs).
        ser_prediction = np.argmax(predictions[0]) # emotion index
        
        ser_prediction = str(ser_prediction) # int type data is not JSON serializable and 
                                             # must be converted to string
        
    return ser_prediction


