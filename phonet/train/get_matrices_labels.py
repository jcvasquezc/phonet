
import sys
import os
from six.moves import cPickle as pickle
import numpy as np
PATH=os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH)
from Phonological import Phonological
from tqdm import tqdm

if __name__=="__main__":
    if len(sys.argv)!=4:
        print("python get_matrices_labels.py <path_features> <path_labels> <path_seq_out>")
        sys.exit()

    Phon=Phonological()
    path_feat=sys.argv[1]
    path_lab=sys.argv[2]
    path_seq=sys.argv[3]
    if not os.path.exists(path_seq):
        os.makedirs(path_seq)
    len_seq=40

    hf=os.listdir(path_lab)
    hf.sort()

    hf2=os.listdir(path_feat)
    hf2.sort()

    Feat=[]
    warnings=0
    pbar=tqdm(range(len(hf)))
    for j in pbar:
        pbar.set_description("Processing %s" % hf[j])
        pickle_file1=path_feat+hf[j]
        if not (hf[j] in hf2):
            print("warning, file labels not found in audio", hf[j])
            warnings=warnings+1
            continue
        pickle_file2=path_lab+hf[j]
        with open(pickle_file1, 'rb') as f:
            feat = pickle.load(f)
        f.close()
        with open(pickle_file2, 'rb') as f:
            dict1, dict2 = pickle.load(f)
        f.close()

        nf=int(feat.shape[0]/len_seq)
        start=0
        fin=len_seq
        for r in range(nf):
            Lab=Phon.get_d2()

            featmat_t=feat[start:fin,:]
            keyslab=Lab.keys()
            for k in keyslab:
                Lab[k]=dict2[k][start:fin]
            start=start+len_seq
            fin=fin+len_seq

            list_phonokeys=list(Phon.get_list_phonological())
            list_phonokeys.append("phoneme_code")

            for k in list_phonokeys:
                Lab[k]=np.stack(Lab[k], axis=0)
                Lab[k]=np.expand_dims(Lab[k], axis=2)
            

            save={'features': featmat_t, 'labels':Lab}
            file_lab=path_seq+hf[j].replace('.pickle', '')+'_'+str(r)+'.pickle'

            try:
                f = open(file_lab, 'wb')
                pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
                f.close()
            except Exception as e:
                print('Unable to save data to', file_lab, ':', e)