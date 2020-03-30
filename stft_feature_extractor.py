import librosa
import os
import numpy as np
import noisereduce as nr
#import matplotlib.pyplot as plt

def save_STFT(file, name, activity, subject):
    # read audio data
    audio_data, sample_rate = librosa.load(file)
    
    trimmed, index = librosa.effects.trim(audio_data, top_db=20, frame_length=512, hop_length=64)
    
    # extract features
    stft = np.abs(librosa.stft(trimmed, n_fft=512, hop_length=256, win_length=512))
    
    # save features   
    np.save("STFT_features/stft_257_1/" + subject + "_" + name[:-4] + "_" + activity + ".npy", stft)

#2 types of events present in audio
activities = ['tring','calling']
#"s01" for training, "s02" for testing   
subjects = ['s01','s02']

for activity in activities:
    #print(activity)
    for subject in subjects:
        #print(subject)
        innerDir = subject + "/" + activity
        for file in os.listdir(innerDir):
            #print(file)
            if(file.endswith(".mp3")):
                save_STFT(innerDir + "/" + file, file, activity, subject)
                print(subject,activity,file)
            elif(file.endswith(".wav")):
                save_STFT(innerDir + "/" + file, file, activity, subject)
                print(subject,activity,file)







