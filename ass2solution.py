# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 18:20:41 2021

@author: thiag
"""


DATASET_PATH = "music_speech"
BLOCK_SIZE = 1024
HOP_SIZE = 256
SAMPLE_RATE = 44100


import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os


def block_audio(x,blockSize,hopSize,fs):
    
    inLen = len(x)
    nBlock = int(np.ceil((inLen-blockSize)/hopSize)+1)

    xb = np.zeros((nBlock,blockSize))
    timeInSample = np.arange(0, hopSize*nBlock, hopSize)
    timeInSec = timeInSample/fs
                  
    for i in range(len(timeInSec)):
        if i == len(timeInSec)-1:
            zeroPad = blockSize - len(x[int(timeInSample[i]):])
            xb[i] = np.pad(x[int(timeInSample[i]):], (0,zeroPad))
        else:
            xb[i] = x[int(timeInSample[i]):int(timeInSample[i]+blockSize)]
            
    # revised version covered the rest samples
    # inLen = len(x)
    # nBlock = int(np.ceil(inLen / hopSize))
    
    # xb = np.zeros((nBlock, blockSize))
    # timeInSample = np.arange(0, hopSize*nBlock, hopSize)
    # timeInSec = timeInSample/fs
    
    # for i in range(len(timeInSec)):
    #     if blockSize > x[i*hopSize:].size:
    #         zeroPad = blockSize - len(x[int(timeInSample[i]):])
    #         xb[i] = np.pad(x[int(timeInSample[i]):], (0,zeroPad))
    #     else:
    #         xb[i] = x[int(timeInSample[i]):int(timeInSample[i]+blockSize)]
  
    return [xb, timeInSec]


def extract_spectral_centroid(xb, fs):
    
    [nBlocks_,blockSize_] = xb.shape
    specCentroidVector = np.zeros(nBlocks_)

    # Spectral Centroid in Hz, compute from the magnitude spectrum (not power spectrum)

    Hann = np.hanning(blockSize_)
    freqIdx = np.fft.fftfreq(2*blockSize_,1/fs)[:blockSize_]

    for idx, val in enumerate(xb):
        
         freqMag = np.abs(np.fft.fft(val*Hann,2*blockSize_))[:blockSize_]
         specCentroidVector[idx] = np.sum(freqMag*freqIdx)/np.sum(freqMag)

    return specCentroidVector


def extract_rms(xb):
    
    [nBlocks_,blockSize_] = xb.shape
    RMSvector = np.zeros(nBlocks_)
    
    # RMS in dB, truncated at -100dB
    
    for idx, val in enumerate(xb):
        RMSvector[idx] = np.sqrt(np.mean(val**2))
        
    RMSvector = 20 * np.log10(RMSvector)
    RMSvector[RMSvector < -100] = -100
        
    return RMSvector


def extract_zerocrossingrate(xb):
    
    [nBlocks_,blockSize_] = xb.shape
    ZCRateVector = np.zeros(nBlocks_)
    
    def sign(x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0
    
    ZCAux = np.zeros(blockSize_)
    for idx, val in enumerate(xb):
        for SampIdx, Sample in enumerate(val):
            ZCAux[SampIdx] = sign(Sample)
            
        ZCRateVector[idx] = np.mean(np.abs(np.diff(ZCAux)))/2
        
    return ZCRateVector 


def extract_spectral_crest(xb):
    
    [nBlocks_,blockSize_] = xb.shape
    SpecCrestVector = np.zeros(nBlocks_)

    Hann = np.hanning(blockSize_)
    Spectrogram = np.abs(np.fft.fft(xb*Hann,2*blockSize_))
    Spectrogram_ = Spectrogram[:,:blockSize_]
    
    maxSpec = Spectrogram_.max(axis=1)
    Spectrogram_[Spectrogram_== 0] = 1
    
    SpecCrestVector = maxSpec/sum(Spectrogram_.T)
    
    return SpecCrestVector 


def extract_spectral_flux(xb):
    
    [nBlocks_,blockSize_] = xb.shape
    SpecFluxVector = np.zeros(nBlocks_)

    Hann = np.hanning(blockSize_)
    Spectrogram = np.abs(np.fft.fft(xb*Hann,2*blockSize_))
    Spectrogram_ = Spectrogram[:,:blockSize_]
    Spectrogram_1 = np.concatenate((np.zeros((1,blockSize_)),Spectrogram_),axis=0)
    
    SpecDiff = np.diff(Spectrogram_1, axis=0)
    SpecFluxVector = np.sqrt(np.sum(SpecDiff**2, axis=1))/(blockSize_/2)
            
    return SpecFluxVector 


def extract_features(x, blockSize, hopSize, fs):
    
    [xb, timeInSec] = block_audio(x,blockSize,hopSize,fs)
    [nBlocks_,blockSize_] = xb.shape
    featuresMatrix = np.zeros((5, nBlocks_))
    
    SC = extract_spectral_centroid(xb, fs)
    RMS = extract_rms(xb)
    ZCR = extract_zerocrossingrate(xb)
    SCR = extract_spectral_crest(xb)
    SF = extract_spectral_flux(xb)

    
    featuresMatrix[0] = SC
    featuresMatrix[1] = RMS
    featuresMatrix[2] = ZCR
    featuresMatrix[3] = SCR
    featuresMatrix[4] = SF
    print("featuresMatrix", featuresMatrix.shape)

    return featuresMatrix


def aggregate_feature_perfile(featuresPerFile):
    aggFeaMatrix = np.zeros((10, 1))
    aggFeaMatrix[0,0] = np.mean(featuresPerFile[0])
    aggFeaMatrix[1,0] = np.std(featuresPerFile[0])
    aggFeaMatrix[2,0] = np.mean(featuresPerFile[1])
    aggFeaMatrix[3,0] = np.std(featuresPerFile[1])
    aggFeaMatrix[4,0] = np.mean(featuresPerFile[2])
    aggFeaMatrix[5,0] = np.std(featuresPerFile[2])
    aggFeaMatrix[6,0] = np.mean(featuresPerFile[3])
    aggFeaMatrix[7,0] = np.std(featuresPerFile[3])
    aggFeaMatrix[8,0] = np.mean(featuresPerFile[4])
    aggFeaMatrix[9,0] = np.std(featuresPerFile[4])
    print("aggFeaMatrix", aggFeaMatrix.shape)
    return aggFeaMatrix


def get_feature_data(path, blockSize, hopSize):
    featureDataMatrix = np.zeros((10, 0))
    mapping = []
    for root, dirs, filenames in os.walk(path):
        for file in filenames:
            mapping.append(os.path.join(root, file))
    fileNumber = 1
    for wav_path in mapping:
        if os.path.splitext(wav_path)[1] == ".wav":
            print("---------Evaluating file", fileNumber, "-----------")
            sampleRate, audio = wavfile.read(wav_path)
            print(wav_path, ": success read!")
            features = extract_features(audio, blockSize, hopSize, fs)
            aggFeatures = aggregate_feature_perfile(features)
            featureDataMatrix = np.concatenate((featureDataMatrix, aggFeatures), axis=1)
            fileNumber = fileNumber + 1
    print("featureDataMatrix", featureDataMatrix.shape)
    return featureDataMatrix


def normalize_zscore(featureData):
    mean = np.ones(featureData.shape[0])
    std = np.ones(featureData.shape[0])
    normFeatures = featureData.copy()
    for i in range(featureData.shape[0]):
        mean[i] = np.mean(featureData[i])
        std[i] = np.std(featureData[i])
        normFeatures[i] = (featureData[i] - mean[i]) / std[i]
    print("normFeatures", normFeatures.shape)
    return normFeatures


def draw(x_var_music, y_var_music, x_var_speech, y_var_speech, x, y):
   
    plt.subplots()
    ax1 = plt.scatter(x_var_music, y_var_music, c='r')
    ax2 = plt.scatter(x_var_speech, y_var_speech, c='b')
    # plt.xscale('log')
    # plt.yscale('log')
    title = '{}/{} Feature Comparison'.format(x, y)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)

    plt.legend((ax1, ax2), ('music', 'speech'),
               numpoints=1, loc='upper right', ncol=3, fontsize=8)
    plt.grid(False)

    plt.show()


def visualize_features(path_to_musicspeech):
    
    for root, dirs, filenames in os.walk(path_to_musicspeech):
        for directory in dirs:
            if directory == "music_wav":
                music_path = os.path.join(root, directory)
            elif directory == "speech_wav":
                speech_path = os.path.join(root,directory)
                
    music_featureData = get_feature_data(music_path, blockSize, hopSize)
    music_splitPoint = music_featureData.shape[1]
    
    speech_featureData = get_feature_data(speech_path, blockSize, hopSize)
    speech_splitPoint = speech_featureData.shape[1]
    
    combined_featureData = np.concatenate((music_featureData, speech_featureData), 1)
    normFeatureMatrix = normalize_zscore(combined_featureData)
    normMusic_featureData = normFeatureMatrix[:,0:music_splitPoint]
    normSpeech_featureData = normFeatureMatrix[:,-speech_splitPoint:]
    
    m_SC_mean = normMusic_featureData[0,:]
    s_SC_mean = normSpeech_featureData[0,:]
    m_SCR_mean = normMusic_featureData[6,:]
    s_SCR_mean = normSpeech_featureData[6,:]
    draw(m_SC_mean, m_SCR_mean, s_SC_mean, s_SCR_mean, 'Mean SC', 'Mean SCR')

    m_SF_mean = normMusic_featureData[8,:]
    s_SF_mean = normSpeech_featureData[8,:]
    m_ZCR_mean = normMusic_featureData[4,:]
    s_ZCR_mean = normSpeech_featureData[4,:]
    draw(m_SF_mean, m_ZCR_mean, s_SF_mean, s_ZCR_mean, 'Mean SF', 'Mean ZCR')
    
    m_RMS_mean = normMusic_featureData[2,:]
    s_RMS_mean = normSpeech_featureData[2,:]
    m_RMS_std = normMusic_featureData[3,:]
    s_RMS_std = normSpeech_featureData[3,:]
    draw(m_RMS_mean,m_RMS_std, s_RMS_mean, s_RMS_std, 'Mean RMS', 'Std RMS')
    
    m_ZCR_std = normMusic_featureData[5,:]
    s_ZCR_std = normSpeech_featureData[5,:]
    m_SCR_std = normMusic_featureData[7,:]
    s_SCR_std = normSpeech_featureData[7,:]
    draw(m_ZCR_std, m_SCR_std, s_ZCR_std, s_SCR_std, 'Std ZCR', 'Std SCR')
    
    m_SC_std = normMusic_featureData[1,:]
    s_SC_std = normSpeech_featureData[1,:]
    m_SF_std = normMusic_featureData[9,:]
    s_SF_std = normSpeech_featureData[9,:]
    draw(m_SC_std, m_SF_std, s_SC_std, s_SF_std, 'Std SC', 'Std SF')
  
    return [normMusic_featureData, normSpeech_featureData]


if __name__ == '__main__':
    
    blockSize = BLOCK_SIZE
    hopSize = HOP_SIZE
    dataPath = DATASET_PATH
    fs = SAMPLE_RATE

    f1 = 441
    f2 = 882
    t0 = 0
    t1 = 1
    t2 = 2
    
    samples1 = t1*fs
    samples2 = (t2-t1)*fs
    dur1 = np.arange(t0,t1,1/fs)
    dur2 = np.arange(t1,t2,1/fs)
    y1 = np.sin(2 * np.pi * f1 * dur1)
    y2 = np.sin(2 * np.pi * f2 * dur2)
    y3 = ((np.random.rand(10000)/100000)-0.1)
    
    y = np.concatenate((y1,y2,y3), axis=None)
    dur = np.concatenate((dur1,dur2), axis=None)
    
    windoewdY = y * np.hanning(len(y))
    
    sp = np.fft.rfft(y,2*len(y))[:int(np.ceil(y.shape[-1]))]
    freq = np.fft.fftfreq(2*len(y),1/fs)[:int(np.ceil(y.shape[-1]))]
    plt.plot(freq[:5000],np.abs(sp)[:5000])
    
    x=y
    [xb, timeInSec] = block_audio(x,blockSize,hopSize,fs)
    
    SC = extract_spectral_centroid(xb, fs)
    SF = extract_spectral_flux(xb)
    SCR = extract_spectral_crest(xb)
    ZCR = extract_zerocrossingrate(xb)
    RMS = extract_rms(xb)
    
    [normMusic_featureData, normSpeech_featureData] = visualize_features(dataPath)
    