import numpy as np
import librosa
#import matplotlib.pyplot as plt
import noisereduce as nr
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
#import IPython
import os
import pyaudio
import statistics

#Load segment audio classification model
model_path = r"Models/"
model_name = "audio_NN_New2020_03_24_23_29_12_acc_50.0"

# Model reconstruction from JSON file
with open(model_path + model_name + '.json', 'r') as f:
	model = model_from_json(f.read())

# Load weights into the new model
model.load_weights(model_path + model_name + '.h5')

# Replicate label encoder
lb = LabelEncoder()
lb.fit_transform(['tring', 'calling'])

#Some Utils
def minMaxNormalize(arr):
	mn = np.min(arr)
	mx = np.max(arr)
	return (arr-mn)/(mx-mn)
    
def predictSoundEvent(clip):
    
	stfts = np.abs(librosa.stft(clip, n_fft=512, hop_length=256, win_length=512))
	stfts = np.mean(stfts,axis=1)
	stfts = minMaxNormalize(stfts)
	result = model.predict(np.array([stfts]))
	
	if result[0][1] > 1e-10: # 1e-15:
		label = "tring"
	else:
		label = "calling"
    
	
	return label
    

# Split a given long audio file on silent parts.
# Accepts audio numpy array audio_data, window length w and hop length h, threshold_level, tolerence
# threshold_level: Silence threshold
# Higher tolence to prevent small silence parts from splitting the audio.
# Returns array containing arrays of [start, end] points of resulting audio clips
def split_audio(audio_data, w, h, threshold_level, tolerence=10):
	split_map = []
	start = 0
	data = np.abs(audio_data)
	#threshold = threshold_level*np.mean(data[:25000])
	threshold=threshold_level
	inside_sound = False
	near = 0
	  
	for i in range(0,len(data)-w, h):
		win_mean = np.mean(data[i:i+w])
		
		if(win_mean>threshold and not(inside_sound)):
			inside_sound = True
			start = i
		if(win_mean<=threshold and inside_sound and near>tolerence):#indexes are not appended even if mean<threshold, untill "tolerance" no. of such cases have been observed
			inside_sound = False
			near = 0
			split_map.append([start, i])
		if(inside_sound and win_mean<=threshold):
			near += 1
	return split_map

CHUNKSIZE = 22050 # fixed chunk size
RATE = 22050 # sampling rate

# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

#noise window
data = stream.read(10000)
noise_sample = np.frombuffer(data, dtype=np.float32)

audio_buffer = []
near = 0
label="calling"
inside_tring = False


while(True):
	# Read chunk and load it into numpy array.
	data = stream.read(CHUNKSIZE)
	current_window = np.frombuffer(data, dtype=np.float32)
    
	#Reduce noise real-time
	#current_window = nr.reduce_noise(audio_clip=current_window, noise_clip=noise_sample, verbose=False)
	current_window = current_window
	
    
	if(audio_buffer==[]):
		audio_buffer = current_window
	else:
		
		if(near<2):
			audio_buffer = np.concatenate((audio_buffer,current_window))
			near += 1
		else:
			nr_audio=np.array(audio_buffer)
			nr_audio1 = nr_audio - statistics.mean(nr_audio) # dc blocking
			
			#passing absolute threshold to split_audio
			sound_clips = split_audio(nr_audio1, 10000, 2500, 0.002, 3)
			#print(sound_clips)

			if(sound_clips==[]):
				label="calling"
				if(label == "calling" and inside_tring):
					print("tring stop")
					inside_tring=False				

			for intvl in sound_clips:
				clip, index = librosa.effects.trim(nr_audio1[intvl[0]:intvl[1]], top_db=20, frame_length=512, hop_length=64) # Empherically select top_db for every sample
				
				label = predictSoundEvent(clip)
				if(label == "tring" and not(inside_tring)): #event "tring" has arrived and previously we were not in "tring" zone
					print("tring start")
					inside_tring=True
				elif(label == "calling" and inside_tring): #event "calling" has arrived and previously we were in "tring" zone
					print("tring stop")
					inside_tring=False
                
                
			audio_buffer = []
			near = 0
    

# close stream
stream.stop_stream()
stream.close()
p.terminate()



