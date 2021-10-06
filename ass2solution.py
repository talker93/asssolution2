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
    SpecCentroidVector = np.zeros((nBlocks_,1))

    # Do the mathemagic
    # Spectral Centroid in Hz, compute from the magnitude spectrum (not power spectrum)
    return SpecCentroidVector


def extract_rms(xb, fs):
    
    [nBlocks_,blockSize_] = xb.shape
    RMSvector = np.zeros((nBlocks_,1))
    
    # Do the mathemagic
    # RMS in dB, truncated at -100dB
    return RMSvector

def extract_zerocrossingrate(xb, fs):
    
    [nBlocks_,blockSize_] = xb.shape
    ZCRateVector = np.zeros((nBlocks_,1))
    
    # Do the mathemagic
    return ZCRateVector 

def extract_spectral_crest(xb, fs):
    
    [nBlocks_,blockSize_] = xb.shape
    SpecCrestVector = np.zeros((nBlocks_,1))

    # Do the mathemagic
    return SpecCrestVector 

def extract_spectral_flux(xb, fs):
    
    [nBlocks_,blockSize_] = xb.shape
    SpecFluxVector = np.zeros((nBlocks_,1))

    # Do the mathemagic
    return SpecFluxVector 

def extract_features(x, blockSize, hopSize, fs) :
    
    [xb, timeInSec] = block_audio(x,blockSize,hopSize,fs)
    [nBlocks_,blockSize_] = xb.shape
    features = np.zeros((nBlocks_,5))
    
    SpecCentroidVector = extract_spectral_centroid(xb, fs)
    RMSvector = extract_rms(xb, fs)
    ZCRateVector = extract_zerocrossingrate(xb, fs)
    SpecCrestVector = extract_spectral_crest(xb, fs)
    SpecFluxVector = extract_spectral_flux(xb, fs)
    
    

    return features