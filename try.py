import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import os
from scipy.io import wavfile

'''define a function for log mel spectrogram to extract the features 
of audio files and represent it in image.'''
def log_spectrogram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, np.log(spec.T.astype(np.float32) + eps)

'''write a function to convert raw audio files into images of spectrogram '''
def wav2img(figsize=(10, 10)):
    """
    takes in wave file path
    and the fig size. Default 10,10 will make images 161 x 99
    """
    fig = plt.figure(figsize=figsize)    
    '''use soundfile library to read in the wave files'''
    samplerate, test_sound  = wavfile.read(filepath)
    #print(filepath)
    _, spectrogram = log_spectrogram(test_sound, samplerate)
    dir = os.path.basename(os.path.dirname(filepath))
    #print(dir)
    ## create output path
    output_file = os.path.basename(filepath).split('.')[0]
    output_file = SavePath + dir + '/' + output_file + '.png' 
    #print(output_file)
    #plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    plt.imsave(output_file, spectrogram)
    plt.close()

if __name__ == '__main__':
    path = 'C:/Users/mukes/Desktop/TensorFlow Speech Recognition Challenge/train/audio/'
    os.chdir(path)
    folders = os.listdir()
    WavFiles = []
    SavePath = 'C:/Users/mukes/Desktop/TensorFlow Speech Recognition Challenge/picts/train/'
    for i in folders:
        FilePath = path + i
        #print(FilePath)
        for j in os.listdir(FilePath):
            print(j)
            WavFiles.append(FilePath +'/'+ j)
    #print(type(WavFiles[1]))
    # for filepath in WavFiles:
    #     wav2img()