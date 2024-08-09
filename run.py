from flask import Flask, render_template, request, jsonify
import os
from tensorflow.python.keras.models import load_model
import tensorflow as tf
import numpy as np
import librosa
from firebase_config import firebase_initialize

firebase_initialize()

# Load the pre-trained model

model = tf.keras.models.load_model('modelfull.h5', compile=False)

app = Flask(__name__)
app.static_folder = 'static'
# Define the expected input shape of the model
target_sr=16000

# Function to resample audio to the target sampling rate and shape
def resample_audio(audio, target_sr=16000, target_length=16000):
    # Resample audio
    audio_resampled = librosa.resample(audio, orig_sr=len(audio), target_sr=target_sr)
    # Pad or truncate audio to the target length
    if len(audio_resampled) < target_length:
        audio_resampled = np.pad(audio_resampled, (0, target_length - len(audio_resampled)))
    else:
        audio_resampled = audio_resampled[:target_length]
    # Reshape audio to (target_length, 1)
    audio_resampled = audio_resampled.reshape(-1, 1)
    return audio_resampled



# Function to convert audio to FFT
def audio_to_fft(audio):
    fft = tf.signal.fft(tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64))
    fft = tf.expand_dims(fft, axis=-1)
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])

def predict(audio):
    audio = tf.expand_dims(audio, axis=0)  # Add batch dimension
    ffts = audio_to_fft(audio)
    y_pred = model.predict(ffts)
    prediction_index = np.argmax(y_pred)
    print("Shape of y_pred:", y_pred.shape)
    print("Predicted values:", y_pred)
    print("Predicted :", prediction_index)

    if prediction_index == 0:
        prediction = "FAKE"
    else:
        prediction = "REAL"
    return prediction



@app.route('/predict', methods=['POST'])
def predict_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file provided'})
    
    file = request.files['audio']
    print("Received file:", file.filename)  # Debugging
    
    audio, _ = librosa.load(file, sr=target_sr)  # Explicitly set the sample rate to target_sr
    print("Audio shape:", audio.shape)  # Debugging
    
    # Resample the audio to the target sampling rate and shape
    audio = resample_audio(audio)
    print("Processed audio shape:", audio.shape)  # Debugging
    
    prediction = predict(audio) 
    # print("Prediction:", prediction)
    return jsonify({'prediction': prediction})

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/detection',methods=["GET"])
def detection():
    return render_template('detection.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/signup')
def signup():
    return render_template('signup.html')

if __name__ == '__main__':
    app.run(debug=True)

# import tensorflow as tf
# import os
# from tensorflow.python.keras.models import load_model
# import numpy as np
# import librosa
# from flask import Flask, render_template, request, jsonify

# app = Flask(__name__)
# app.static_folder = 'static'

# # Load the pre-trained model
# model = load_model('model2.h5', compile=False)

# # Define the expected input shape of the model
# target_sr = 16000
# target_length = 16000  # Target audio length

# # Function to resample audio to the target sampling rate and shape
# def resample_audio(audio, target_sr=target_sr, target_length=target_length):
#     # Resample audio
#     audio_resampled = librosa.resample(audio, orig_sr=len(audio), target_sr=target_sr)
#     # Pad or truncate audio to the target length
#     if len(audio_resampled) < target_length:
#         audio_resampled = np.pad(audio_resampled, (0, target_length - len(audio_resampled)))
#     else:
#         audio_resampled = audio_resampled[:target_length]
#     # Reshape audio to (target_length, 1)
#     audio_resampled = audio_resampled.reshape(-1, 1)
#     return audio_resampled

# # Function to convert audio to FFT
# def audio_to_fft(audio):
#     fft = tf.signal.fft(tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64))
#     fft = tf.expand_dims(fft, axis=-1)
#     return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])

# # Function to make predictions
# def predict(audio):
#     audio = tf.expand_dims(audio, axis=0)  # Add batch dimension
#     ffts = audio_to_fft(audio)
#     y_pred = model.predict(ffts)
#     prediction_index = np.argmax(y_pred)
#     if prediction_index == 0:
#         prediction = "FAKE"
#     else:
#         prediction = "REAL"
#     return prediction

# # Route for predicting audio
# @app.route('/predict', methods=['POST'])
# def predict_audio():
#     if 'audio' not in request.files:
#         return jsonify({'error': 'No file provided'})
    
#     file = request.files['audio']
#     audio, _ = librosa.load(file, sr=target_sr)  # Explicitly set the sample rate to target_sr
#     audio = resample_audio(audio)  # Resample audio
#     prediction = predict(audio)
#     return jsonify({'prediction': prediction})

# # Home route
# @app.route('/')
# def home():
#     return render_template("home.html")

# # Login route
# @app.route('/login')
# def login():
#     return render_template('login.html')

# # Detection route
# @app.route('/detection', methods=["GET"])
# def detection():
#     return render_template('detection.html')

# # About route
# @app.route('/about')
# def about():
#     return render_template('about.html')

# # Signup route
# @app.route('/signup')
# def signup():
#     return render_template('signup.html')

# if __name__ == '__main__':
#     app.run(debug=True)
