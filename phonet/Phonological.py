
"""
Created on Feb 28 2019
@author: J. C. Vasquez-Correa
        Pattern recognition Lab, University of Erlangen-Nuremberg
        Faculty of Engineering, University of Antioquia,
        juan.vasquez@fau.de
"""

import numpy as np
import pandas as pd

class Phonological:

    def __init__(self):

        self.list_phonological={"vocalic" : ["a","e","i","o","u", "w", "j"],
                      "consonantal" : ["b", "B","d", "D","f", "F","k","l","m","n", "N","p","r","rr","s", "Z", "T","t","g", "G","tS","S","x", "jj", "J", "L", "z"],
                      "back"        : ["a","o","u", "w"],
                      "anterior"    : ["e","i","j"],
                      "open"        : ["a","e","o"],
                      "close"       : ["j","i","u", "w"],
                      "nasal"       : ["m","n", "N"],
                      "stop"        : ["p","b", "B","t","k","g", "G","tS","d", "D"],
                      "continuant"  : ["f", "F","b", "B","tS","d", "D","s", "Z", "T","x", "jj", "J","g", "G","S","L","x", "jj", "J", "z"],
                      "lateral"     :["l"],
                      "flap"        :["r"],
                      "trill"       :["rr"],
                      "voice"       :["a","e","i","o","u", "w","b", "B","d", "D","l","m","n", "N","rr","g", "G","L", "j"],
                      "strident"    :["tS","f", "F","s", "Z", "T", "z",  "S"],
                      "labial"      :["m","p","b", "B","f", "F"],
                      "dental"      :["t","d", "D"],
                      "velar"       :["k","g", "G"],
                      "pause"       :  ["sil", "<p:>"]}

    def get_list_phonological(self):
        return self.list_phonological

    def get_list_phonological_keys(self):
        keys=self.list_phonological.keys()
        return list(keys)


    def get_d1(self):
        keys=self.get_list_phonological_keys()
        dict_1={"xmin":[],"xmax":[],"phoneme":[],"phoneme_code":[]}
        for k in keys:
            dict_1[k]=[]
        return dict_1

    def get_d2(self):
        keys=self.get_list_phonological_keys()
        dict_2={"n_frame":[],"phoneme":[],"phoneme_code":[]}
        for k in keys:
            dict_2[k]=[]
        return dict_2

    def get_list_phonemes(self):
        keys=self.get_list_phonological_keys()
        phon=[]
        for k in keys:
            phon.append(self.list_phonological[k])
        phon=np.hstack(phon)

        return np.unique(phon)


def main():
    phon=Phonological()
    keys=phon.get_list_phonological_keys()

    d1=phon.get_d1()
    d2=phon.get_d2()
    ph=phon.get_list_phonemes()

if __name__=="__main__":
    main()


