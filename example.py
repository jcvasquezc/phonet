# -*- coding: utf-8 -*-
"""
Created on Feb 28 2019
@author: J. C. Vasquez-Correa
        Pattern recognition Lab, University of Erlangen-Nuremberg
        Faculty of Engineering, University of Antioquia,
        juan.vasquez@fau.de
"""

from phonet import Phonet
import os

if __name__=="__main__":

    PATH=os.path.dirname(os.path.abspath(__file__))
    phon=Phonet()

    # get the "stop" phonological posterior from a single file
    file_audio=PATH+"/audios/pataka.wav"
    file_feat=PATH+"/phonclasses/pataka"
    phon.get_phon_wav(file_audio, file_feat, "stop", True)

    # get the "nasal" phonological posterior from a single file
    file_audio=PATH+"/audios/sentence.wav"
    file_feat=PATH+"/phonclasses/sentence_nasal"
    phon.get_phon_wav(file_audio, file_feat, "nasal", True)

    # get the "strident" phonological posterior from a single file
    file_feat=PATH+"/phonclasses/sentence_strident"
    phon.get_phon_wav(file_audio, file_feat, "strident", True)

    # get "all" phonological posteriors from a single file
    file_feat=PATH+"/phonclasses/sentence_all"
    phon.get_phon_wav(file_audio, file_feat, "all", True)

    # compute the posteriorgram for an audio_file
    phon.get_posteriorgram(file_audio)


    # get "all" phonological posteriors from de audio files included in a directory
    directory=PATH+"/phonclasses/"
    phon.get_phon_path(PATH+"/audios/", PATH+"/phonclasses2/", "all")

    
