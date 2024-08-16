import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

SAMPLE_RATE = 22050 #podana w datasecie
TRACK_DURATION = 30 #sekundy
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

model = keras.models.load_model('musicGenreRecogniser.keras')

file_path = input("Path to wav file:")
if file_path == "":
    file_path = "genres/blues/blues.00000.wav"


if not os.path.isfile(file_path):
    print("File not found")
elif model is None:
    print("Model not found")
else:
    try:
        signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512).T
        mfcc = np.expand_dims(mfcc, axis=0)
        y = model.predict([mfcc])
        print(genres[np.argmax(y)])
    except:
        print("Invalid file format")

    # signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
    # mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512).T
    # mfcc = np.expand_dims(mfcc, axis=0)
    # y = model.predict([mfcc])
    # print(genres[np.argmax(y)])


