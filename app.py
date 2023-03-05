import os
import wave
import joblib
import numpy as np
import pandas as pd
import prediction as predict
from PIL import Image
import streamlit as st
import soundfile as sf
import matplotlib.pyplot as plt
from audiorecorder import audiorecorder
from streamlit_option_menu import option_menu
from sklearn.preprocessing import RobustScaler

image = Image.open(r"emotions.png")
st.sidebar.image(image, caption='Human Emotions')

st.title('ವಾಕ್ಯ ಭಾವ ಪ್ರಕೀರ್ಣ')
st.markdown(":green[A Sanskrit phrase] represented in Kannada. :book:")
st.markdown('#')

def record_audio():
    st.write("Have no audio file? No worries.\nYou can now record your own emotion!")
    audio = audiorecorder("Click to record", "Recording...")
    if len(audio) > 0:
        # To play audio in frontend:
        st.audio(audio.tobytes())
        
        with open('output.dat', 'wb') as output_file:
            # Write the audio data to the file
            output_file.write(audio.tobytes())

        with open('audio.dat', 'rb') as file:
            audio_bytes = file.read()
            audio_ndarray = np.frombuffer(audio_bytes, dtype=np.int16)
            with open('output1.dat', 'wb') as output_file:
            # Write the audio data to the file
                output_file.write(audio.tobytes())

        # Convert bytes to NumPy ndarray
        

        # To save audio to a file:
        wav_file = open(r"audio_rec.wav", "wb")
        wav_file.write(audio.tobytes())
        wav_file.close()

def convert_to_2d(X):
    x = []
    for i in range(len(X)):
        x.append(X[i].flatten())
        
    return np.array(x)

def predict_emotion(path):

    # path = r"D:\App\audio_file.wav"
    mfcc = predict.save_mfcc(path, 'mfcc', num_segments=3)
    mels = predict.save_mfcc(path, 'mels', num_segments=3)
    cqt = predict.save_mfcc(path, 'chroma_cqt', num_segments=3)
    stft = predict.save_mfcc(path, 'chroma_stft', num_segments=3)

    xmels = convert_to_2d(mfcc)
    xmfcc = convert_to_2d(mels)
    xstft = convert_to_2d(stft)
    xcqt = convert_to_2d(cqt)

    xall = np.concatenate((xmels, xmfcc, xstft, xcqt))
    scale = RobustScaler()
    x = scale.fit_transform(xall)

    mlp_model = joblib.load(r"D:\Git Repos\Speech Emotion Recognition\models\mlp_model_rob.joblib")
    predictions = mlp_model.predict(x)
    labels, counts = np.unique(predictions, return_counts=True)

    mapping = {0:'Anger', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
    y = pd.DataFrame(labels, counts).reset_index()
    y.columns = ['count', 'emotion']
    y['emotion'] = y['emotion'].map(mapping)
    y = y.sort_values(by='count', ascending=False)
    emotions = list(y['emotion'].loc[:1])

    return emotions
    
def return_image(image_name):
    image_root_path = 'D:\App\Images'
    path = os.path.join(image_root_path, image_name)
    return Image.open(path)

def return_text(file_name):
    text_root_path = r'D:\App\app_content'
    path = os.path.join(text_root_path, file_name)
    with open(path) as fp:
        text = fp.read()
    return text


selected = option_menu("Home", ["Intro", 'Try it', 'Record', 'About Us'], icons=['triangle', 'gear', 'mic', 'people'], menu_icon="house", default_index=0, orientation="horizontal")

if selected == "Intro":
    st.write(return_text('intro.txt'))
    st.markdown('***')

    option = st.selectbox('Want to visualize how we perceive audio? Check it out', ['Time Domain Representaion', 'Frequncy Domain Representaion', "MFCC", 'Mels'])
    if option == 'MFCC':
        st.image(return_image('mfcc.png'))
    elif option == 'Mels':
        st.image(return_image('mels.png'))
    elif option == 'Time Domain Representaion':
        st.image(return_image('tAE.png'))
    else:
        st.image(return_image('fSC.png'))

elif selected == 'About Us':
    st.write(return_text('about-us.txt'))
    st.markdown('***')
    st.write('Want to get in touch? Lets have some intresting conversation!')
    image = return_image('linkedin-circled.png')

    ro_url = "www.linkedin.com/in/rohit-bc-7a25741b3"
    mahaa_url = "https://www.linkedin.com/in/smrithi-shaji-2b81b9210"
    st.markdown("[Rohit BC](ro_url)")
    st.markdown("[Smrithi Shaji](mahaa_url)")



elif selected == 'Try it':
    st.write('Start with uploading your audio file')
    audio_file = st.file_uploader('', type=["wav", "mp3", "ogg"])
    if audio_file:
        with open("audio_file.wav", "wb") as f:
            f.write(audio_file.getvalue())
        st.success("Audio file saved!")

    path = r"D:\App\Audio Files\audio_file.wav"
    emotions = predict_emotion(path)
    st.title(f"Predicted Emotions are '{emotions[0]}',  '{emotions[1]}'")

elif selected == 'Record':
    record_audio()
    st.write('Working on it... Hopefully tomorrow we will complete!')
