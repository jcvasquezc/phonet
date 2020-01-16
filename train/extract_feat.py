
import sys
import os
from scipy.io.wavfile import read
import numpy as np
import pysptk.sptk as sptk
from six.moves import cPickle as pickle

import python_speech_features as pyfeat
from tqdm import tqdm

def extract_feat(signal,fs):
    size_frame=0.025
    time_shift=0.010
    order=13
    nfilt=33
    signal=signal-np.mean(signal)
    signal=signal/np.max(np.abs(signal))
    Fbank, energy=pyfeat.fbank(signal,samplerate=fs,winlen=size_frame,winstep=time_shift,
      nfilt=nfilt,nfft=512,lowfreq=0,highfreq=None,preemph=0.97)
    energy= np.expand_dims(energy, axis=1)
    feat2=np.concatenate((Fbank,energy),axis=1)
    feat2=np.log10(feat2)

    return feat2


if __name__=="__main__":
    if len(sys.argv)!=3:
        print("python extract_feat.py <path_audios> <path_features>")
        sys.exit()

    path_audios=sys.argv[1]
    path_features=sys.argv[2]

    hf=os.listdir(path_audios)
    hf.sort()
    pbar=tqdm(range(len(hf)))
    for j in pbar:
        pbar.set_description("Processing %s" % hf[j])
        fs,data=read(path_audios+hf[j])
        feat=extract_feat(data,fs)
        file_results=path_features+hf[j].replace(".wav", ".pickle")
        try:
            f = open(file_results, 'wb')
            pickle.dump(feat, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', file_results, ':', e)

