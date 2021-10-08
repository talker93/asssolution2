# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 18:20:41 2021

@author: thiag
"""

import numpy as np
from scipy import signal
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


def extract_rms(xb, fs):
    
    [nBlocks_,blockSize_] = xb.shape
    RMSvector = np.zeros(nBlocks_)
    
    # RMS in dB, truncated at -100dB
    
    for idx, val in enumerate(xb):
        RMSvector[idx] = np.sqrt(np.mean(val**2))
        
    RMSvector = 20 * np.log10(RMSvector)
    RMSvector[RMSvector < -100] = -100
        
    return RMSvector

def extract_zerocrossingrate(xb, fs):
    
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

def extract_spectral_crest(xb, fs):
    
    [nBlocks_,blockSize_] = xb.shape
    SpecCrestVector = np.zeros(nBlocks_)

    Hann = np.hanning(blockSize_)
    Spectrogram = np.abs(np.fft.fft(xb*Hann,2*blockSize_))    
    Spectrogram_ = (Spectrogram.T[:blockSize_]).T
    
    maxSpec = Spectrogram_.max(axis=1)
    Spectrogram_[Spectrogram_== 0] = 1
    
    SpecCrestVector = maxSpec/sum(Spectrogram_.T)
    
    return SpecCrestVector 

def extract_spectral_flux(xb, fs):
    
    [nBlocks_,blockSize_] = xb.shape
    SpecFluxVector = np.zeros(nBlocks_)

    Hann = np.hanning(blockSize_)
    Spectrogram = np.abs(np.fft.fft(xb*Hann,2*blockSize_))
    Spectrogram_ = (Spectrogram.T[:blockSize_]).T
    Spectrogram_1 = np.concatenate((np.zeros((blockSize_,1)).T,Spectrogram_),axis=0)
    
    SpecDiff = np.diff(Spectrogram_1, axis=0)
    SpecFluxVector = np.sqrt(np.sum(SpecDiff**2, axis=1))/(blockSize_/2)
            
    
    return SpecFluxVector 

def extract_features(x, blockSize, hopSize, fs):
    
    [xb, timeInSec] = block_audio(x,blockSize,hopSize,fs)
    [nBlocks_,blockSize_] = xb.shape
    features = np.zeros((nBlocks_,5))
    
    SC = extract_spectral_centroid(xb, fs)
    RMS = extract_rms(xb, fs)
    ZCR = extract_zerocrossingrate(xb, fs)
    SCrest = extract_spectral_crest(xb, fs)
    SF = extract_spectral_flux(xb, fs)

    
    features.T[0] = SC
    features.T[1] = RMS
    features.T[2] = ZCR
    features.T[3] = SCrest
    features.T[4] = SF

    return features


fs = 44100
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
blockSize = 256
hopSize = 128
[xb, timeInSec] = block_audio(x,blockSize,hopSize,fs)

SC = extract_spectral_centroid(xb, fs)
SF = extract_spectral_flux(xb, fs)
SCrest = extract_spectral_crest(xb, fs)
ZCR = extract_zerocrossingrate(xb, fs)
RMS = extract_rms(xb, fs)

features = extract_features(x, blockSize, hopSize, fs)