# -*- coding: utf-8 -*-
"""
Compute posteriors probabilities of phonological classes from audio files for several groups of phonemes according to the mode and manner of articulation.

@author: J. C. Vasquez-Correa
        Pattern recognition Lab, University of Erlangen-Nuremberg
        Faculty of Engineering, University of Antioquia,
        juan.vasquez@fau.de
"""

import os
import numpy as np
import python_speech_features as pyfeat
from scipy.io.wavfile import read
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
from matplotlib.patches import Rectangle, Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from scipy.signal import resample_poly

from keras.layers import Input, BatchNormalization, Bidirectional, GRU, Permute, Dropout, Dense, TimeDistributed
from keras.utils import np_utils
from keras.models import Model
from keras import optimizers
import gc
from keras import backend as K
from matplotlib import cm
try:
    from phonet.Phonological import Phonological
except:
    from Phonological import Phonological

from tqdm import tqdm

class Phonet:

    """
    Phonet computes posteriors probabilities of phonological classes from audio files for several groups of phonemes.

    :param phonological_classes: phonological class to be evaluated ("consonantal", "back", "anterior", "open", "close", "nasal", "stop",
                                                "continuant",  "lateral", "flap", "trill", "voice", "strident",
                                                "labial", "dental", "velar", "pause", "vocalic", "all").
    :returns: Phonet Object (see `Examples <https://github.com/jcvasquezc/phonet/blob/master/example.py>`_).

    phonological_classes=='all' computes the phonological posterior for the complete list of phonological classes. 

    """

    def __init__(self, phonological_classes):

        self.path=os.path.dirname(os.path.abspath(__file__))
        self.Phon=Phonological()
        self.phonemes=self.Phon.get_list_phonemes()
        self.GRU_size=128
        self.hidden_size=128
        self.lr=0.0001
        self.recurrent_droput_prob=0.0
        self.size_frame=0.025
        self.time_shift=0.01
        self.nfilt=33
        self.len_seq=40
        self.names=self.Phon.get_list_phonological_keys()
        self.num_labels=num_labels=[2 for j in range(len(self.names))]
        self.nfeat=34
        self.thrplot=0.7
        self.nphonemes=len(self.phonemes)
        if phonological_classes[0]=="all":
            self.keys_val=self.names
        else:
            self.keys_val=phonological_classes
        self.models=self.load_model()
        self.model_phon=self.load_model_phon()
        self.MU, self.STD=self.load_scaler()
        
    def load_model(self):
        Models=[]
        input_size=(self.len_seq, self.nfeat)
        model_file=self.path+"/models/model.h5"
        Model=self.model(input_size)
        Model.load_weights(model_file)
        return Model

    def mask_correction(self, posterior, threshold=0.5):
        """Implements a mask for a correction the posterior probabilities

        :param posterior: phonological posterior.
        :param threshold: threshold for correction
        :returns: Corrected phonological posterior.
        """
        for j in np.arange(1,len(posterior)-1):
            if (posterior[j-1]>=threshold) and (posterior[j]<threshold) and (posterior[j+1]>=threshold):
                posterior[j]=(posterior[j-1]+posterior[j+1])/2
            if (posterior[j-1]<threshold) and (posterior[j]>=threshold) and (posterior[j+1]<threshold):
                posterior[j]=(posterior[j-1]+posterior[j+1])/2
        return posterior



    def load_model_phon(self):
        input_size=(self.len_seq, self.nfeat)
        Model_phonemes=self.path+"/models/phonemes.hdf5"
        Model_phon=self.modelp(input_size)
        Model_phon.load_weights(Model_phonemes)
        return Model_phon

    def load_scaler(self):
        file_mu=self.path+"/models/mu.npy"
        file_std=self.path+"/models/std.npy"
        MU=np.load(file_mu)
        STD=np.load(file_std)

        return MU, STD


    def modelp(self, input_size):
        """This is the architecture used for phoneme recognition
        It consists of a 2 Bidirectional GRU layers, followed by a time-distributed dense layer

        :param input_size: size of input for the BGRU layers (number of features x sequence length).
        :returns: A Keras model of a 2-layer BGRU neural network.
        """
        input_data=Input(shape=(input_size))
        x=input_data
        x=BatchNormalization()(x)
        x=Bidirectional(GRU(self.GRU_size, recurrent_dropout=self.recurrent_droput_prob, return_sequences=True, reset_after=False))(x)
        x=Bidirectional(GRU(self.GRU_size, recurrent_dropout=self.recurrent_droput_prob, return_sequences=True, reset_after=False))(x)
        x = TimeDistributed(Dense(self.hidden_size, activation='relu'))(x)
        x = TimeDistributed(Dense(self.nphonemes, activation='softmax'))(x)
        modelGRU=Model(inputs=input_data, outputs=x)
        opt=optimizers.Adam(lr=self.lr)
        modelGRU.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return modelGRU


    def model(self, input_size):
        """This is the architecture used for the estimation of the phonological classes using a multitask learning strategy
        It consists of a 2 Bidirectional GRU layers, followed by a time-distributed dense layer

        :param input_size: size of input for the BGRU layers (number of features x sequence length).
        :returns: A Keras model of a 2-layer BGRU neural network.
        """
        input_data=Input(shape=(input_size))
        x=input_data
        x=BatchNormalization()(x)
        x=Bidirectional(GRU(self.GRU_size, recurrent_dropout=self.recurrent_droput_prob, return_sequences=True, reset_after=False))(x)
        x=Bidirectional(GRU(self.GRU_size, recurrent_dropout=self.recurrent_droput_prob, return_sequences=True, reset_after=False))(x)
        x=Dropout(0.2)(x)
        x = TimeDistributed(Dense(self.hidden_size, activation='relu'))(x)
        x=Dropout(0.2)(x)
            # multi-task
        xout=[]
        out=[]
        for j in range(len(self.names)):
            xout.append(TimeDistributed(Dense(self.hidden_size, activation='relu'))(x))
            out.append(TimeDistributed(Dense(2, activation='softmax'), name=self.names[j])(xout[-1]))

        modelGRU=Model(inputs=input_data, outputs=out)
        opt=optimizers.Adam(lr=self.lr)
        alphas=list(np.ones(len(self.names))/len(self.names))
        modelGRU.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'], sample_weight_mode="temporal", loss_weights=alphas)
        return modelGRU

    def get_feat(self, signal, fs):
        """
        This method extracts log-Mel-filterbank energies used as inputs
        of the model.

        :param signal: the audio signal from which to compute features. Should be an N array.
        :param fs: the sample rate of the signal we are working with, in Hz.
        :returns: A numpy array of size (NUMFRAMES by 33 log-Mel-filterbank energies) containing features. Each row holds 1 feature vector.
        """
        signal=signal-np.mean(signal)
        signal=signal/np.max(np.abs(signal))
        mult = int(fs*self.size_frame*self.len_seq)
        fill = int(self.len_seq*self.time_shift*fs)
        fillv=0.05*np.random.randn(fill)
        signal=np.hstack((signal,fillv))
        Fbank, energy=pyfeat.fbank(signal,samplerate=fs,winlen=self.size_frame,winstep=self.time_shift,
          nfilt=self.nfilt,nfft=512,lowfreq=0,highfreq=None,preemph=0.97)

        energy= np.expand_dims(energy, axis=1)
        feat2=np.concatenate((Fbank,energy),axis=1)
        return np.log10(feat2)

    def number2phoneme(self, seq):
        """
        Converts the prediction of the neural network for phoneme recognition to a list of phonemes.

        :param seq: sequence of integers obtained from the preiction of the neural network for phoneme recognition.
        :returns: A list of strings of the phonemes recognized for each time-frame.
        """

        try:
            phonemes=[self.phonemes[j] for j in seq]

            for j in range(1,len(phonemes)-1):
                if phonemes[j]!=phonemes[j-1] and phonemes[j]!=phonemes[j+1]:
                    phonemes[j]=phonemes[j-1]

            return phonemes
        except:
            return np.nan


    def get_phon_wav(self, audio_file, feat_file="", plot_flag=True):
        """
        Estimate the phonological classes using the BGRU models for an audio file (.wav)

        :param audio_file: file audio (.wav) sampled at 16 kHz
        :param feat_file: . File (.csv) to save the posteriors for the phonological classes. Deafult="" does not save the csv file
        :param plot_flag: True or False, whether you want plots of phonological classes or not
        :returns: A pandas dataFrame with the posterior probabilities for the phonological classes.
        
        >>> from phonet.phonet import Phonet
        >>> phon=Phonet(["stop"]) # get the "stop" phonological posterior from a single file
        >>> file_audio=PATH+"/audios/pataka.wav"
        >>> file_feat=PATH+"/phonclasses/pataka"
        >>> phon.get_phon_wav(file_audio, file_feat, True)

        >>> file_audio=PATH+"/audios/sentence.wav"
        >>> file_feat=PATH+"/phonclasses/sentence_nasal"
        >>> phon=Phonet(["nasal"]) # get the "nasal" phonological posterior from a single file
        >>> phon.get_phon_wav(file_audio, file_feat, True)

        >>> file_audio=PATH+"/audios/sentence.wav"
        >>> file_feat=PATH+"/phonclasses/sentence_nasal"
        >>> phon=Phonet(["strident", "nasal", "back"]) # get "strident, nasal, and back" phonological posterior from a single file
        >>> phon.get_phon_wav(file_audio, file_feat, True)

        """
        if audio_file.find('.wav')==-1 and audio_file.find('.WAV')==-1:
            raise ValueError(audio_file+" is not a valid audio file")

        fs, signal=read(audio_file)
        if fs!=16000:
            #raise ValueError(str(fs)+" is not a valid sampling frequency")
            signal=resample_poly(signal, 16000, fs)
            fs=16000
        feat=self.get_feat(signal,fs)

        nf=int(feat.shape[0]/self.len_seq)

        start=0
        fin=self.len_seq
        Feat=[]
        for j in range(nf):
            featmat_t=feat[start:fin,:]
            Feat.append(featmat_t)
            start=start+self.len_seq
            fin=fin+self.len_seq
        Feat=np.stack(Feat, axis=0)
        Feat=Feat-self.MU
        Feat=Feat/self.STD
        df={}
        dfa={}
        pred_mat_phon=np.asarray(self.model_phon.predict(Feat))
        pred_mat_phon_seq=np.concatenate(pred_mat_phon,0)
        pred_vec_phon=np.argmax(pred_mat_phon_seq,1)

        nf=int(len(signal)/(self.time_shift*fs)-1)
        if nf>len(pred_vec_phon):
            nf=len(pred_vec_phon)
        
        phonemes_list=self.number2phoneme(pred_vec_phon[:nf])

        t2=np.arange(nf)*self.time_shift
        
        df["time"]=t2
        df["phoneme"]=phonemes_list
        dfa["time"]=t2
        dfa["phoneme"]=phonemes_list

        pred_mat=np.asarray(self.models.predict(Feat))

        
        for l, problem in enumerate(self.keys_val):

            index=self.names.index(problem)
            pred_matv=pred_mat[index][:,:,1]
            post=np.hstack(pred_matv)[:nf]
            dfa[problem]=self.mask_correction(post)
        
        if plot_flag:
            n_plots=int(np.ceil(len(self.keys_val)/4))
            figsize=(6,int(n_plots*3))
            colors = cm.get_cmap('Accent', 5)
            col_order=[0,1,2,3]*n_plots
            plt.figure(figsize=figsize)
        for l, problem in enumerate(self.keys_val):

            df[problem]=dfa[problem]

            if plot_flag:
                
                if (l==0) or (l==4) or (l==8) or (l==12) or (l==16):
                    subp=int(l/4+1)
                    plt.subplot(n_plots,1, subp)
                    t=np.arange(len(signal))/fs
                    signal=signal-np.mean(signal)
                    plt.plot(t,signal/np.max(np.abs(signal)), color=colors.colors[4], alpha=0.5)
                    plt.grid()

                plt.plot(t2,df[problem],  color=colors.colors[col_order[l]], label=problem, linewidth=2)
                ini=t2[0]
                for nu in range(1,len(phonemes_list)):
                    if phonemes_list[nu]!=phonemes_list[nu-1] or nu==len(phonemes_list)-1:
                        difft=t2[nu]-ini
                        plt.text(x=ini+difft/2, y=1, s=phonemes_list[nu-1], color="k", fontsize=10)
                        ini=t2[nu]

                plt.xlabel("Time (s)")
                plt.ylabel("Phonological posteriors")
                plt.legend(loc=8, ncol=2)

        if plot_flag:
            plt.tight_layout()
            plt.savefig(feat_file+"post.png")
            plt.show()

        df2=pd.DataFrame(df)
        if len(feat_file)>0:
            df2.to_csv(feat_file)
        gc.collect()
        return df2




    def get_phon_path(self, audio_path, feat_path, plot_flag=False):
        """
        Estimate the phonological classes using the BGRU models for all the (.wav) audio files included inside a directory

        :param audio_path: directory with (.wav) audio files inside, sampled at 16 kHz
        :param feat_path: directory were the computed phonological posteriros will be stores as a (.csv) file per (.wav) file from the input directory
        :param plot_flag: True or False, whether you want plots of phonological classes or not
        :returns: A directory with csv files created with the posterior probabilities for the phonological classes.

        >>> from phonet.phonet import Phonet
        >>> phon=Phonet(["vocalic", "strident", "nasal", "back", "stop", "pause"])
        >>> phon.get_phon_path(PATH+"/audios/", PATH+"/phonclasses2/")
        """

        hf=os.listdir(audio_path)
        hf.sort()

        if not os.path.exists(feat_path):
            os.makedirs(feat_path)

        if feat_path[-1]!="/":
            feat_path=feat_path+"/"

        pbar=tqdm(range(len(hf)))

        for j in pbar:
            pbar.set_description("Processing %s" % hf[j])
            audio_file=audio_path+hf[j]
            feat_file=feat_path+hf[j].replace(".wav", ".csv")
            self.get_phon_wav(audio_file, feat_file, plot_flag)


    def get_posteriorgram(self, audio_file):
        """
        Estimate the posteriorgram for an audio file (.wav) sampled at 16kHz

        :param audio_file: file audio (.wav) sampled at 16 kHz
        :returns: plot of the posteriorgram

        >>> from phonet.phonet import Phonet
        >>> phon=Phonet(["vocalic", "strident", "nasal", "back", "stop", "pause"])
        >>> phon.get_posteriorgram(file_audio)
        """
        if audio_file.find('.wav')==-1 and audio_file.find('.WAV')==-1:
            raise ValueError(audio_file+" is not a valid audio file")

        fs, signal=read(audio_file)
        if fs!=16000:
            raise ValueError(str(fs)+" is not a valid sampling frequency")
        feat=self.get_feat(signal,fs)
        nf=int(feat.shape[0]/self.len_seq)
        start=0
        fin=self.len_seq
        Feat=[]
        for j in range(nf):
            featmat_t=feat[start:fin,:]
            Feat.append(featmat_t)
            start=start+self.len_seq
            fin=fin+self.len_seq
        Feat=np.stack(Feat, axis=0)
        Feat=Feat-self.MU
        Feat=Feat/self.STD

        pred_mat_phon=np.asarray(self.model_phon.predict(Feat))
        pred_mat_phon_seq=np.concatenate(pred_mat_phon,0)
        pred_vec_phon=np.argmax(pred_mat_phon_seq,1)
        nf=int(len(signal)/(self.time_shift*fs)-1)
        phonemes_list=self.number2phoneme(pred_vec_phon[:nf])
        t=np.arange(nf)*self.time_shift
        posteriors=[]
        pred_mat=np.asarray(self.models.predict(Feat))
        for l, problem in enumerate(self.keys_val):
            
            index=self.names.index(problem)
            pred_matv=pred_mat[index][:,:,1]
            post=np.hstack(pred_matv)[:nf]
            posteriors.append(self.mask_correction(post))

        posteriors=np.vstack(posteriors)
        plt.figure()
        plt.imshow(np.flipud(posteriors), extent=[0, t[-1], 0, len(self.keys_val)], aspect='auto')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Phonological class")
        plt.yticks(np.arange(len(self.keys_val))+0.5, self.keys_val)
        ini=t[0]
        for nu in range(1,len(phonemes_list)):
            if phonemes_list[nu]!=phonemes_list[nu-1] or nu==len(phonemes_list)-1:
                difft=t[nu]-ini
                plt.text(x=ini+difft/2, y=19, s="/"+phonemes_list[nu-1]+"/", color="k", fontsize=12)
                ini=t[nu]
        plt.colorbar()
        plt.show()


    def get_PLLR(self, audio_file, feat_file="", projected=True, plot_flag=False):
        """
        Estimate the phonological log-likelihood ratio (PLLR) features for an audio file (.wav) sampled at 16kHz
        
        :param audio_file: file audio (.wav) sampled at 16 kHz
        :param feat_file: .csv file  to save the PLLR features for the phonological classes. Deafult="" does not save the csv file
        :projected: whether to make a projection of the feature space of the PLLR according to [1], in order to avoid the bounding effect.
        :plot_flag: True or False. Plot distributions of the feature space
        :returns: Pandas dataFrame with the PLLR features

        >>> from phonet.phonet import Phonet
        >>> phon=Phonet(["all"])
        >>> file_audio=PATH+"/audios/sentence.wav"
        >>> phon.get_PLLR(file_audio)

        References:

        [1] Diez, M., Varona, A., Penagarikano, M., Rodriguez-Fuentes, L. J., & Bordel, G. (2014). On the projection of PLLRs for unbounded feature distributions in spoken language recognition. IEEE Signal Processing Letters, 21(9), 1073-1077.

        [2] Abad, A., Ribeiro, E., Kepler, F., Astudillo, R. F., & Trancoso, I. (2016). Exploiting Phone Log-Likelihood Ratio Features for the Detection of the Native Language of Non-Native English Speakers. In INTERSPEECH (pp. 2413-2417).
        """

        df=self.get_phon_wav(audio_file, plot_flag=plot_flag)
        dfPLLR={}
        dfPLLR["time"]=df["time"]
        PLLR=np.zeros((len(df["time"]), len(self.keys_val)))
        post=np.zeros((len(df["time"]), len(self.keys_val)))
        
        for l, problem in enumerate(self.keys_val):

            PLLR[:,l]=np.log10(df[problem]/(1-df[problem]))
            post[:,l]=df[problem]
        if projected:
            N=PLLR.shape[1]
            I=np.identity(N)
            Ones=np.ones((N,N))*1/np.sqrt(N)
            P=I-Ones.T*Ones
            PLLRp=np.matmul(PLLR,P)

            if plot_flag:
                figsize=(10,3)
                
                fig=plt.figure(figsize=figsize)
                
                ax = fig.add_subplot(131, projection='3d')
                indexes=np.random.randint(0,PLLR.shape[1], 3)
                ax.scatter(post[:,indexes[0]],post[:,indexes[1]], post[:,indexes[2]], c='b', alpha=0.2, s=20)
                ax.set_xlabel(self.keys_val[indexes[0]])
                ax.set_ylabel(self.keys_val[indexes[1]])
                ax.set_zlabel(self.keys_val[indexes[2]])
                plt.title("Posteriors")
                ax = fig.add_subplot(132, projection='3d')
                ax.scatter(PLLR[:,indexes[0]],PLLR[:,indexes[1]],PLLR[:,indexes[2]], c='b', alpha=0.2, s=20)
                ax.set_xlabel(self.keys_val[indexes[0]])
                ax.set_ylabel(self.keys_val[indexes[1]])
                ax.set_zlabel(self.keys_val[indexes[2]])
                plt.title("PLLR")
                ax = fig.add_subplot(133, projection='3d')
                ax.scatter(PLLRp[:,indexes[0]],PLLRp[:,indexes[1]],PLLRp[:,indexes[2]], c='b', alpha=0.2, s=20)
                ax.set_xlabel(self.keys_val[indexes[0]])
                ax.set_ylabel(self.keys_val[indexes[1]])
                ax.set_zlabel(self.keys_val[indexes[2]])
                plt.title("Projected PLLR")
                plt.tight_layout()
        
        for l, problem in enumerate(self.keys_val):
            if projected:
                dfPLLR[problem]=PLLRp[:,l]
            else:
                dfPLLR[problem]=PLLR[:,l]

        if plot_flag:
            ncols=2
            nrows=int(np.ceil(len(self.keys_val)/4))
            figsize=(6,int(nrows*4))
            colors = cm.get_cmap('Accent', 4)
            colorsn=[0,1,2,3]*len(self.keys_val)
            plt.figure(figsize=figsize)
            for l, problem in enumerate(self.keys_val):

                if (l==0) or (l==4) or (l==8) or (l==12) or (l==16):
                    subp1=int(2*l/4+1)
                    subp2=int(2*l/4+2)
                plt.subplot(nrows,ncols, subp1)
                plt.hist(df[problem], color=colors.colors[colorsn[l]], label=problem, alpha=0.5)
                if l==len(self.keys_val)-1:
                    plt.xlabel("Phonological posteriors")
                plt.subplot(nrows,ncols, subp2)
                plt.hist(dfPLLR[problem], color=colors.colors[colorsn[l]], label=problem, alpha=0.5)
                if l==len(self.keys_val)-1:
                    plt.xlabel("PLLR")
                plt.legend()
                plt.tight_layout()
                plt.grid()
            plt.show()

        dfPLLR=pd.DataFrame(dfPLLR)
        if len(feat_file)>0:
            dfPLLR.to_csv(feat_file)
        return dfPLLR
