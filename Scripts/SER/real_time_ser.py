"""
	
SPEECH EMOTION RECOGNITION
	
"""

# Import necessary libraries
import numpy as np
import librosa
import noisereduce as nr
import time
import struct
import wave
import pyaudio
from pydub import AudioSegment, effects
from keras.models import load_model
from array import array

# Load h5 model
model = load_model("/home/pi/project/codes/ser_model.h5")

# Definition of function to perform preprocess of audio files to 3D arrays.
def preprocess(file_path, frame_length = 2048, hop_length = 512):
    
    '''
    A process to an audio .wav file before execcuting a prediction.
      Arguments:
      - file_path - The system path to the audio file.
      - frame_length - Length of the frame over which to compute the speech features. default: 2048
      - hop_length - Number of samples to advance for each frame. default: 512

      Return:
        'X_3D' variable, containing a shape of: (batch, timesteps, feature) for a single file (batch = 1).
    ''' 
    
    # Fetch sample rate
    _, sr = librosa.load(path = file_path, sr = None)
    
    # Load audio file
    rawsound = AudioSegment.from_file(file_path, duration = None) 
    
    # Normalize to 5 dBFS 
    normalizedsound = effects.normalize(rawsound, headroom = 5.0) 
    
    # Transform the audio file to np.array of samples
    normal_x = np.array(normalizedsound.get_array_of_samples(), dtype = 'float32') 
    
    # Noise reduction                  
    final_x = nr.reduce_noise(normal_x, sr=sr)
        
    # Features extraction of RMS, ZCR and MFCC  
    f1 = librosa.feature.rms(y=final_x, frame_length=frame_length, hop_length=hop_length, center=True, pad_mode='reflect').T 
    f2 = librosa.feature.zero_crossing_rate(y=final_x, frame_length=frame_length, hop_length=hop_length, center=True).T
    f3 = librosa.feature.mfcc(y=final_x, sr=sr, S=None, n_mfcc=13, hop_length = hop_length).T 
    X = np.concatenate((f1, f2, f3), axis = 1)
    
    # Save features in a 3D array to input to the model
    X_3D = np.expand_dims(X, axis=0)
    
    print(f1)
    print('----------')
    print(f2)
    print('----------')
    print(f3)
    print('----------')
    print(X_3D)
    
    return X_3D, f1

# Definition of the emotion dict.
emotions = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'sad', 5 : 'surprise', 6 : 'neutral'}
emo_list = list(emotions.values())

# Definition of function to detect if there is "silence".
def is_silent(data_rms):
    
    # Compute average RMS value
    avg_rms = np.mean(data_rms)
    
    # Check if the average RMS value is below the 'silent' threshold
    return avg_rms < 18000000

"........................................................................................................................................"

# Real time detection

# Initialize variables
RATE = 48000 # samples/s
CHUNK = 512 # hop length
RECORD_SECONDS = 3.61

FORMAT = pyaudio.paInt32
CHANNELS = 1
WAVE_OUTPUT_FILE = "/home/pi/project/codes/output.wav"

# Open an input channel
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=1)

# Initialize a non-silent signals array to state "True" in the first 'while' iteration
data = array('h', np.random.randint(size = 512, low = 0, high = 500))

# Start a session with the list of recordings
print("** session started")
total_predictions = [] # A list for all predictions in the session.
tic = time.perf_counter()

# If there is not silence, start recording
while is_silent(data) == False:
    print("* recording...")
    frames = [] 
    data = np.nan # Reset 'data' variable.

    timesteps = int(RATE / CHUNK * RECORD_SECONDS) # => 339

    # Insert frames to 'output.wav'.
    for i in range(0, timesteps):
        data = array('l', stream.read(CHUNK)) 
        frames.append(data)

        wf = wave.open(WAVE_OUTPUT_FILE, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print("* done recording")
    
    # Preprocess WAV file
    x = preprocess(WAVE_OUTPUT_FILE) # 'output.wav' file preprocessing.
    
    # Model's prediction => 7 emotion probabilities array.
    predictions = model.predict(x, use_multiprocessing=True)
    pred_list = list(predictions)
    pred_np = np.squeeze(np.array(pred_list).tolist(), axis=0) # Get rid of 'array' & 'dtype' statments.
    total_predictions.append(pred_np)
    
    # Present emotion distribution for a sequence (3.61 secs).
    max_emo = np.argmax(predictions)
    print('max emotion:', emotions.get(max_emo,-1))
    
    print(100*'-')
    
    # Define the last 2 seconds sequence.
    last_frames = np.array(struct.unpack(str(96 * CHUNK) + 'B' , np.stack(( frames[-1], frames[-2], frames[-3], frames[-4],
                                                                            frames[-5], frames[-6], frames[-7], frames[-8],
                                                                            frames[-9], frames[-10], frames[-11], frames[-12],
                                                                            frames[-13], frames[-14], frames[-15], frames[-16],
                                                                            frames[-17], frames[-18], frames[-19], frames[-20],
                                                                            frames[-21], frames[-22], frames[-23], frames[-24]),
                                                                            axis =0)) , dtype = 'b')
    # If the last 2 seconds are silent, end the session.
    if is_silent(last_frames): 
        break

# Session end      
toc = time.perf_counter()
stream.stop_stream()
stream.close()
p.terminate()
wf.close()
print('** session ended')
print(f"Emotions analyzed for: {(toc - tic):0.4f} seconds")

"""

Bibliography: 
    https://github.com/MeidanGR/SpeechEmotionRecognition_Realtime
	
"""

