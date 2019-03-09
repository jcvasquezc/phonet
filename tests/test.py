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

import matplotlib
matplotlib.use('agg')


if __name__=="__main__":

    PATH=os.path.dirname(os.path.abspath(__file__))
    phon=Phonet()

    file_audio=PATH+"/../audios/pataka.wav"
    file_feat=PATH+"/../phonclasses/pataka"
    phon.getphonclass(file_audio, file_feat, "stop", False)

    file_audio=PATH+"/../audios/sentence.wav"

    file_feat=PATH+"/../phonclasses/sentence_nasal"
    phon.getphonclass(file_audio, file_feat, "nasal", False)

    file_feat=PATH+"/../phonclasses/sentence_strident"
    phon.getphonclass(file_audio, file_feat, "strident", False)

    file_feat=PATH+"/../phonclasses/sentence_all"
    phon.getphonclass(file_audio, file_feat, "all", False)
