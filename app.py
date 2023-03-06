import os
import wave
import joblib
import numpy as np
import pandas as pd
import prediction as predict
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import soundfile as sf
import matplotlib.pyplot as plt
from audiorecorder import audiorecorder
from streamlit_option_menu import option_menu
from sklearn.preprocessing import RobustScaler

global_flag = 0
image = Image.open(r"Images\emotions.png")
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

    flag = 0
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
    predictions_ = mlp_model.predict(x)
    labels, counts = np.unique(predictions_, return_counts=True)

    mapping = {0:'Anger', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

    predictions = pd.DataFrame(labels, counts).reset_index()
    predictions.columns = ['count', 'emotion']
    predictions['emotion'] = predictions['emotion'].map(mapping)
    predictions = predictions.sort_values(by='count', ascending=False)

    file_path1 = return_file_path(predictions['emotion'][0])
    emotion1 = predictions['emotion'][0]
    try:
        file_path2 = return_file_path(predictions['emotion'][1])
        emotion2 = predictions['emotion'][1]
    except:
        flag = 1
        global_flag = 1

    if flag == 1:
        return file_path1, file_path2

    return file_path1, file_path2, emotion1, emotion2
    
def return_file_path(emotion):
    if emotion == 'Anger':
        return r'Images\angry.png'
    elif emotion == 'Fear':
        return r'Images\fear.png'
    elif emotion == 'Happy':
        return r'Images\happy.png'    
    elif emotion == 'Sad':
        return r'Images\sad.png'    
    elif emotion == 'Neutral':
        return r'Images\neutral.png'    
    elif emotion == 'Disgust':
        return r'Images\disgust.png'
    

def image_file(file_path, emotion, image_size=[300, 280], font_path='Roboto-Black.ttf', rgb_colors=(255, 0, 0), font_size=65, cordinates=(510, 875)):
    img = Image.open(file_path)
    I1 = ImageDraw.Draw(img)
    myFont = ImageFont.truetype(r"Fonts\Roboto-Black.ttf", font_size)
    I1.text(cordinates, emotion, fill=rgb_colors, font=myFont)
    img = img.resize(image_size)
    return img


def plot_emotions(file_path1, file_path2, emotion1, emotion2, rows=1, columns=2, figsize=(10, 5)):
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(rows, columns, 1)
    image = image_file(file_path1, emotion1)
    plt.imshow(image)
    plt.axis('off')
    
    fig.add_subplot(rows, columns, 2)
    image = image_file(file_path2, emotion2)
    plt.imshow(image)
    plt.axis('off')
    
    plt.show()


def return_image(image_name):
    image_root_path = 'D:\App\Images'
    path = os.path.join(image_root_path, image_name)
    return Image.open(path)


def return_text(file_name):
    text_root_path = r'app_content'
    path = os.path.join(text_root_path, file_name)
    with open(path) as fp:
        text = fp.read()
    return text


selected = option_menu("Home", ["Intro", 'Try it', 'About Us'], icons=['triangle', 'gear', 'people'], menu_icon="house", default_index=0, orientation="horizontal")

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
        with open(r"Audio Files\audio_file.wav", "wb") as f:
            f.write(audio_file.getvalue())
        st.success("Audio file saved!")

    path = r"Audio Files\audio_file.wav"

    if global_flag == 1:
        file_path1, emotion1 = predict_emotion(path)
        st.image([file_path1], width=280)
        st.markdown("<h3 style='text-align: center; color: red;'>{}</h3>".format(emotion1), unsafe_allow_html=True)
    else:
        file_path1, file_path2, emotion1, emotion2 = predict_emotion(path)
        st.image([file_path1, file_path2], width=280)
        st.markdown("<h3 style='text-align: center; color: red;'>{}, {}</h3>".format(emotion1, emotion2), unsafe_allow_html=True)


elif selected == 'Record':
    record_audio()
    st.write('Working on it... Hopefully tomorrow we will complete!')
