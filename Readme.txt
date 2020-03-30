AUTHOR: AVIGYAN SINHA

OBJECTIVE: To identify audio event corresponding to the occurrence of the sound "tring tring" (that is the ringing of a phone) in real-time from the stream coming from a microphone. The program "realtime_sound_classification_from_mic.py" detects the start and end of "tring" sound in the microphone feed and prints "tring start" and "tring stop" respectively in the Linux terminal.

The submission folder contains the following items(decription given below):

1) Models - This folder contains the trained neural network model files

2) STFT_features - This folder contains the extracted STFT features of the audio files, used to train the neural network

3) s01 - This folder contains 2 subfolders(calling, tring) each containing 20 audio files, respectively, corresponding to the 2 classes(calling, tring) that the neural network is trained on

4) s02 - This folder comtains 2 subfolders(calling, tring) each containing 1 audio file, respectively, corresponding to the 2 classes(calling, tring) that the neural network is tested on

5) sound_clips - This folder contains 4 example files have combinations of "tring", ringtone and human conversation

6) requirements.txt - This list the libraries required to run the python files in this project

7) stft_feature_extractor.py - This file extracts the STFT features from the audio files in "s01", "s02" folders

8) sound_event_classifier.py - This file trains the neural network architecture using the STFT features and saves the model in "Models" folder

9) realtime_sound_classification_from_mic.py - This file detects the start and end of "tring" in real-time audio from microphone, using the trained neural network model


USAGE: python3 realtime_sound_classification_from_mic.py
(All programs were run using Python 3.6.9)





