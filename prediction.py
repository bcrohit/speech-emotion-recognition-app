import warnings
warnings.filterwarnings('ignore')

import os
import math
import json
import librosa
import numpy as np


class AudioFeatureExtractor:
    def __init__(self, audio_path, samples_per_segment, num_mfcc_vectors_per_segment, vectors, 
                 num_segments=3, num_mfcc=40, num_mels=40, n_fft=2048, hop_length=512):
        
        self.file_path = audio_path
        self.signal, self.sr = librosa.load(audio_path)
        self.samples_per_segment = samples_per_segment
        self.num_mfcc_vectors_per_segment = num_mfcc_vectors_per_segment
        self.num_segments = num_segments
        self.num_mfcc = num_mfcc
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_length = hop_length
        self.vectors = vectors


class MFCCExtractor(AudioFeatureExtractor):
    def extract(self):
        for segment in range(self.num_segments):        
            start = self.samples_per_segment * segment
            finish = start + self.samples_per_segment
            
            mfcc = librosa.feature.mfcc(y=self.signal[start:finish], sr=self.sr, n_mfcc=self.num_mfcc, 
                                        n_fft=self.n_fft, hop_length=self.hop_length)
            
            mfcc = mfcc.T
            
            if len(mfcc) == self.num_mfcc_vectors_per_segment:
                self.vectors["mfcc"].append(mfcc.tolist())
                # print("{}, segment:{}".format(self.file_path, segment+1))



class MelSpectrogramExtractor(AudioFeatureExtractor):
    def extract(self):
        for segment in range(self.num_segments):
            start = self.samples_per_segment * segment
            finish = start + self.samples_per_segment

            mel_spec = librosa.feature.melspectrogram(y=self.signal[start:finish], sr=self.sr, 
                                                      n_fft=self.n_fft, n_mels=self.num_mels, hop_length=self.hop_length)
            
            log_mel_spec = librosa.power_to_db(mel_spec).T

            if len(log_mel_spec) == self.num_mfcc_vectors_per_segment:
                self.vectors['mels'].append(log_mel_spec.tolist())
                # print("{}, segment:{}".format(self.file_path, segment+1))


class STFTExtractor(AudioFeatureExtractor):
    def extract(self):
        for segment in range(self.num_segments):
            start = self.samples_per_segment * segment
            finish = start + self.samples_per_segment
            
            chroma_stft = librosa.feature.chroma_stft(y=self.signal[start:finish], sr=self.sr, n_chroma=self.num_mels, 
                                                      n_fft=self.n_fft, hop_length=self.hop_length)
            
            log_chroma_stft = librosa.power_to_db(chroma_stft).T
            
            if len(log_chroma_stft) == self.num_mfcc_vectors_per_segment:
                self.vectors["chroma_stft"].append(log_chroma_stft.tolist())
                # print("{}, segment:{}".format(self.file_path, segment+1))



class CQTExtractor(AudioFeatureExtractor):
    def extract(self):
        for segment in range(self.num_segments):
            start = self.samples_per_segment * segment
            finish = start + self.samples_per_segment
            try:
                chroma_cqt = librosa.feature.chroma_cqt(y=self.signal[start:finish], sr=self.sr, 
                                                        n_chroma=self.num_mels, bins_per_octave=80)
                
                log_chroma_cqt = librosa.power_to_db(chroma_cqt).T

                if len(log_chroma_cqt) == self.num_mfcc_vectors_per_segment:
                    self.vectors["chroma_cqt"].append(log_chroma_cqt.tolist())
                    # print("{}, segment:{}".format(self.file_path, segment+1))
            except: 
                pass


def create_feature_extractor(feature:str):
    if feature == "mfcc":
        return MFCCExtractor
    if feature == "mels":
        return MelSpectrogramExtractor
    if feature == "chroma_cqt":
        return CQTExtractor
    if feature == "chroma_stft":
        return STFTExtractor


sample_rate = 22050
duration = 3 #in seconds
samples_per_signal = sample_rate * duration
    
    
# Driver Function
def save_mfcc(file, feature, num_segments=3, hop_length=512):
    
    vectors = { feature:[] } 
    
    samples_per_segment = int(samples_per_signal / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    
    print(f"Extracting Feature: {feature.upper()}")
    extractor = create_feature_extractor(feature)(file, samples_per_segment, num_mfcc_vectors_per_segment, vectors)
    extractor.extract()
                
    return np.array(vectors[feature])