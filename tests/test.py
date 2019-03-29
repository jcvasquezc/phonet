# -*- coding: utf-8 -*-
"""
Created on Feb 28 2019
@author: J. C. Vasquez-Correa
        Pattern recognition Lab, University of Erlangen-Nuremberg
        Faculty of Engineering, University of Antioquia,
        juan.vasquez@fau.de
"""


import os
import matplotlib
matplotlib.use('agg')
import sys
PATH=os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH+"/../")

from phonet import Phonet

if __name__=="__main__":

    PATH=os.path.dirname(os.path.abspath(__file__))
    phon=Phonet()

    file_audio=PATH+"/../audios/sentence.wav"
    file_feat=PATH+"/../phonclasses/sentence_all"
    phon.get_phon_wav(file_audio, file_feat, "all", False)

    directory=PATH+"/phonclasses/"
    phon.get_phon_path(PATH+"/../audios/", PATH+"/phonclasses2/", "all")

    phon.get_posteriorgram(file_audio)
