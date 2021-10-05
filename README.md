# asssolution2

Features or descriptors are important building blocks of many music analysis systems. In this assignment you will be implementing some of these featuers and in a later assignment you will use these for a classification task. You will also learn how to normalize these features across a dataset and also visualize these features against one another.

Fixes and clarifications are marked red.

General Instructions:

use the provided function declarations and/or function headers. Submitting modified function headers will result in point deductions.
ensure input and output dimensions of your functions are accurate
all vectors and matrices returned have to be np arrays, no lists etc.
no thirdparty modules except numpy, scipy, and matplotlib are allowed, plus either os or glob
when asked for discussion and plot, submit them with the corresponding question number in one pdf
all plot axes must be labeled and the plots easily understandable
submit all your functions defined in one file: ass2solution.py
DO NOT change file names or function names as that will break the test scripts and will result in point deductions
DO NOT plagiarize. Any similarity with publicly available code or between different submissions will result in 0 points for the complete assignment and, if significant, might be reported.
A. Feature Extraction 
[30] Implement functions for 5 audio features: extract_spectral_centroid(xb, fs), extract_rms(xb), extract_zerocrossingrate(xb), extract_spectral_crest(xb), extract_spectral_flux(xb). xb is a matrix of blocked audio data (dimension NumOfBlocks X blockSize, see assignment 1), fs is the sample rate. Implement each of these functions in the default formulation from the text book/slides. Note that for the spectral features, you have to apply a window function to each block. Use the hann window from a previous in-class exercise for this purpose. To avoid feature definition ambiguities:
Spectral Centroid in Hz, compute from the magnitude spectrum (not power spectrum)
RMS in dB, truncated at -100dB
all others as defined in the book
[5] Implement a function: [features] = extract_features(x, blockSize, hopSize, fs) This function serves as a wrapper for feature extraction. It will take a single-channel audio vector x as input, blocks the audio with the given block size and hop size using block_audio(), calls each of the feature extractor functions implemented above and returns a 5 X NumOfBlocks dimensional feature matrix.
[5] Implement a function: [aggFeatures] = aggregate_feature_per_file(features). This function aggregates the feature matrix returned by extract_features(), aggregates the features across blocks using mean and standard deviation and returns a  10x1 aggregated feature matrix.
[10] Implement a  function: [featureData] = get_feature_data(path, blockSize, hopSize) This function loops over all files contained within a folder that is pointed to by the path argument. The function will then successively call the extract_features() and aggregate_feature_per_file() functions and return a 10xN feature matrix of which contains the aggregated features for N audio files contained in the directory.
B. Feature Normalization
[20] Implement a  function: [normFeatureMatrix] = normalize_zscore(featureData) This function applies the z-score normalization scheme to the input feature matrix. The z-score normalization normalizes each feature to a zero mean and unit standard deviation across the entire dataset. Be careful with the dimensions.
C. Feature Visualization
Download the music/speech dataset from here (链接到外部网站。). Unzip it and use the 'music_wav' and 'speech_wav' directories for the rest of the assignment.

[10] Implement a function visualize_features(path_to_musicspeech) that extracts 2 separate feature matrices for the files in each of the folder (music and speech). You will call your get_feature_data() function for that with blockSize = 1024 and hopSize = 256. You will also normalize the feature matrices over the entire dataset. Note that this means you have to normalize both feature matrices with the same z-score. (hint: concatenate the two feature matrices appropriately before calling normalize_zscore())
[20] Within the same function write code that generates plots to visualize the feature space of the following 5 pairs of features: [SC mean, SCR mean], [SF mean, ZCR mean], [RMS mean, RMS std], [ZCR std, SCR std], [SC std, SF std] Note that you want to plot the data points corresponding to music with a different color than the points that correspond to speech. Use red for music and blue for speech. This type of visualization is referred to as scatter plot. What can you infer from these scatter plots?
 
