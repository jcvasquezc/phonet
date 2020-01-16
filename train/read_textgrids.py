
import sys
import os
from scipy.io.wavfile import read
import numpy as np
import pysptk.sptk as sptk
from six.moves import cPickle as pickle
import pandas as pd
PATH=os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH)
from Phonological import Phonological
from tqdm import tqdm


def phoneme2list_phonological(phoneme):
    Phon=Phonological()
    list_phonological=Phon.get_list_phonological()    
    list_phon_values=np.zeros(len(list_phonological))
    keys=list(list_phonological.keys())
    for j in range(len(list_phonological.keys())):
        if phoneme in list_phonological[keys[j]]:
            list_phon_values[j]=1
    return list_phon_values,list_phonological


def phoneme2number(phoneme):
    Phon=Phonological()
    list_phonemes=Phon.get_list_phonemes()
    number=np.where(list_phonemes==phoneme)[0]
    if len(number)>0:
        return number
    else:
        print("phoneme:*"+ phoneme+"*is not in the list")
        sys.exit()
        return np.nan


def read_textgrid(list_textgrid):
    time_shift=0.01
    pos_start_phonemes=list_textgrid.index("item[3]:")
    n_phonemes=int(list_textgrid[pos_start_phonemes+5].replace("intervals:size=",""))
    list_phonemes=[list_textgrid[j] for j in range(pos_start_phonemes+6,len(list_textgrid))]
    Phon=Phonological()
    dict_1=Phon.get_d1()
    dict_2=Phon.get_d2()

    for j in range(1,n_phonemes+1):
        pos_phoneme=list_phonemes.index("intervals["+str(j)+"]:")
        xmin_line=list_phonemes[pos_phoneme+1]
        dict_1["xmin"].append(float(xmin_line.replace("xmin=","")))
        xmax_line=list_phonemes[pos_phoneme+2]
        dict_1["xmax"].append(float(xmax_line.replace("xmax=","")))
        phoneme_line=list_phonemes[pos_phoneme+3]
        phoneme_=phoneme_line.replace("text=","")
        phoneme=phoneme_.replace('"','')

        dict_1["phoneme"].append(phoneme)
        dict_1["phoneme_code"].append(phoneme2number(phoneme))
        list_phonological,list_keys=phoneme2list_phonological(phoneme)

        keys=list(list_keys.keys())
        for k in range(len(keys)):
            dict_1[keys[k]].append(list_phonological[k])

    start=0.
    n=0
    for j in range(len(dict_1["xmin"])):
        while start<dict_1["xmax"][j]:
            dict_2["n_frame"].append(n)
            dict_2["phoneme"].append(dict_1["phoneme"][j])
            dict_2["phoneme_code"].append(dict_1["phoneme_code"][j])
            for k in range(len(keys)):
                dict_2[keys[k]].append(dict_1[keys[k]][j])
            n=n+1
            start=start+time_shift
    return dict_1, dict_2

if __name__=="__main__":
    if len(sys.argv)!=3:
        print("python read_textgrids.py <path_textgrids> <path_labels>")
        sys.exit()

    path_textgrid=sys.argv[1]
    path_labels=sys.argv[2]

    hf=os.listdir(path_textgrid)
    hf.sort()
    pbar=tqdm(range(len(hf)))
    for j in pbar:
        pbar.set_description("Processing %s" % hf[j])
        f=open(path_textgrid+hf[j], "r")
        data=f.readlines()
        f.close()
        data2=[]
        for k in range(len(data)):
            datat=data[k].replace(" ","")
            data2.append(datat.replace("\n",""))
        list_phon, list_labels=read_textgrid(data2)
        #sys.exit()
        file_results=path_labels+hf[j].replace(".TextGrid", ".pickle")
        try:
            f = open(file_results, 'wb')
            pickle.dump((list_phon, list_labels), f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', file_results, ':', e)
