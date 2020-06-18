
import sys
from six.moves import cPickle as pickle

from keras.layers import Input, BatchNormalization, Bidirectional, GRU, Permute, Reshape, Lambda, Dense, RepeatVector, multiply, TimeDistributed, Dropout, LSTM
from keras.utils import np_utils
from keras.models import Model
from keras import optimizers
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import os
from utils import plot_confusion_matrix
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score
#import pandas as pd

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

from utils import confusion_matrix, get_scaler
from Phonological import Phonological

Phon=Phonological()


def generate_data(directory, batch_size, problem, mu, std, num_labels):
    i = 0
    file_list = os.listdir(directory)
    file_list.sort()
    while True:
        seq_batch = []
        y=[]
        for b in range(batch_size):
            if i == len(file_list):
                i = 0
                np.random.shuffle(file_list)
            with open(directory+file_list[i], 'rb') as f:
                save = pickle.load(f)
            f.close()
            seq_batch.append((save['features']-mu)/std)
            y.append(save['labels'][problem])
            i += 1
        y=np.stack(y, axis=0)
        
        y2=np_utils.to_categorical(y)
        ystack=np.concatenate(y, axis=0)
        ystack=np.hstack(ystack)
        lab, count=np.unique(ystack, return_counts=True)
        class_weights=class_weight.compute_class_weight('balanced', np.unique(y), ystack[0,:])
        weights=np.zeros((y.shape))

        for j in range(len(lab)):
            p=np.where(y==lab[j])
            weights[p]=class_weights[j]

        #print(num_labels, len(lab), y2[:,:,0,:].shape, np.unique(y))
        # if len(lab)<num_labels:
        #     y2=np.zeros((batch_size,ystack.shape[1], num_labels))
        #     for j in np.unique(y):
        #         y2[:,:,j]=1

        if np.max(y)<num_labels-1:
            da=np.zeros((batch_size,y2.shape[1], y2.shape[2], num_labels))
            da[:,:,:,0:np.max(y)+1]=y2
            y2=da


        seq_batch=np.stack(seq_batch, axis=0)
        yield seq_batch, y2[:,:,0,:], weights[:,:,0,0]



def generate_data_test(directory, batch_size, mu, std):
    i = 0
    file_list = os.listdir(directory)
    file_list.sort()
    while True:
        seq_batch = []
        for b in range(batch_size):
            with open(directory+file_list[i], 'rb') as f:
                save = pickle.load(f)
            f.close()
            seq_batch.append((save['features']-mu)/std)
            i+=1
        seq_batch=np.stack(seq_batch, axis=0)
        yield seq_batch

def get_test_labels(directory, batch_size, problem):
    i = 0
    file_list = os.listdir(directory)
    file_list.sort()
    y=[]
    for i in range(len(file_list)):
        with open(directory+file_list[i], 'rb') as f:
            save = pickle.load(f)
        f.close()
        y.append(save['labels'][problem])
    y=np.stack(y, axis=0)
    ystack=np.concatenate(y, axis=0)
    return ystack
    
def DeepArch(input_size, GRU_size, hidden_size, num_labels, Learning_rate, recurrent_droput_prob):
    input_data=Input(shape=(input_size))
    x=input_data
    x=BatchNormalization()(x)
    x=Bidirectional(GRU(GRU_size, recurrent_dropout=recurrent_droput_prob, return_sequences=True))(x)
    x=Bidirectional(GRU(GRU_size, recurrent_dropout=recurrent_droput_prob, return_sequences=True))(x)
    x=Dropout(0.2)(x)
    x = TimeDistributed(Dense(hidden_size, activation='relu'))(x)
    x=Dropout(0.2)(x)
    x = TimeDistributed(Dense(num_labels, activation='softmax'))(x)
    modelGRU=Model(inputs=input_data, outputs=x)
    opt=optimizers.Adam(lr=Learning_rate)
    modelGRU.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'], sample_weight_mode="temporal")
    return modelGRU



if __name__=="__main__":

    if len(sys.argv)!=4:
        print("python main_train_RNN_phoneme.py <path_seq_train> <path_seq_test> <path_results>")
        sys.exit()

    file_feat_train=sys.argv[1]
    file_feat_test=sys.argv[2]
    file_results=sys.argv[3]
    problem="phoneme_code"


    Nfiles_train=len(os.listdir(file_feat_train))
    Nfiles_test=len(os.listdir(file_feat_test))


    if not os.path.exists(file_results):
        os.makedirs(file_results)



    checkpointer = ModelCheckpoint(filepath=file_results+'phonemes_weights.hdf5', verbose=1, save_best_only=True)

    #perc=test_labels(file_feat_test)
    #print("perc_classes=", perc)
    if os.path.exists(file_results+"mu.npy"):

        mu=np.load(file_results+"mu.npy")
        std=np.load(file_results+"std.npy")
    else:
        mu, std=get_scaler(file_feat_train)

        np.save(file_results+"mu.npy", mu)
        np.save(file_results+"std.npy", std)


    phonemes=Phon.get_list_phonemes()
    input_size=(40,34)
    GRU_size=128
    hidden=128
    num_labels=len(phonemes)
    Learning_rate=0.0005
    recurrent_droput_prob=0.0
    epochs=1000
    batch_size=64

    modelPH=DeepArch(input_size, GRU_size, hidden, num_labels, Learning_rate, recurrent_droput_prob)
    print(modelPH.summary())

    steps_per_epoch=int(Nfiles_train/batch_size)
    validation_steps=int(Nfiles_test/batch_size)


    if os.path.exists(file_results+'phonemes_weights.hdf5'):
        modelPH.load_weights(file_results+'phonemes_weights.hdf5')

    earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=0)
    history=modelPH.fit_generator(generate_data(file_feat_train, batch_size, problem, mu, std, num_labels), steps_per_epoch=steps_per_epoch, #workers=4, use_multiprocessing=False,
    epochs=epochs, shuffle=True, validation_data=generate_data(file_feat_test, batch_size, problem, mu, std, num_labels), 
    verbose=1, callbacks=[earlystopper, checkpointer], validation_steps=validation_steps)

    plt.figure()
    plt.plot(np.log(history.history['loss']))
    plt.plot(np.log(history.history['val_loss']))
    plt.xlabel("epochs")
    plt.ylabel("log-Loss")
    plt.savefig(file_results+'Loss.png')
    #plt.show()
    plt.close('all')


    model_json = modelPH.to_json()
    with open(file_results+problem+".json", "w") as json_file:
        json_file.write(model_json)
    try:
        modelPH.save_weights(file_results+problem+'.h5')
    except:
        print('------------------------------------------------------------------------------------------------------------')
        print('┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐')
        print('|                                                                                                           |')
        print('|      FILE  '+file_results+problem+'.h5'+'                                      |')
        print('|             could not be saved                                                                            |')
        print('|                                                                                                           |')
        print('└────────────────────────────────────────────────────────────────────────────────────────────────────────────┘')
        print('------------------------------------------------------------------------------------------------------------')

    np.set_printoptions(precision=4)
    batch_size_val=1
    validation_steps=int(Nfiles_test/batch_size_val)

    ypred=modelPH.predict_generator(generate_data_test(file_feat_test, batch_size_val, mu, std), steps=validation_steps)

    ypredv=np.argmax(ypred, axis=2)
    ypredv=np.concatenate(ypredv, axis=0)

    yt=get_test_labels(file_feat_test, batch_size, problem)
    ytv=np.concatenate(yt,0)
    print(ytv.shape, ypredv.shape, ypred.shape)
    dfclass=classification_report(ytv, ypredv,digits=4)


    print(dfclass)

    class_names=["not "+problem, problem]

    # Plot non-normalized confusion matrix
    #ax1=plot_confusion_matrix(yt, ypredv, classes=class_names,
    #                  title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    ax2=plot_confusion_matrix(ytv, ypredv, file_res=file_results+"/cm.png", classes=class_names, normalize=True,
                        title='Normalized confusion matrix')

    prec=precision_score(ytv, ypredv, average='weighted')
    rec=recall_score(ytv, ypredv, average='weighted')
    f1=f1_score(ytv, ypredv, average='weighted')

    F=open(file_results+"params.csv", "w")
    header="acc_train, acc_dev, loss, val_loss, epochs_run, Fscore, precision, recall\n"
    content=str(history.history["categorical_accuracy"][-1])+", "+str(history.history["val_categorical_accuracy"][-1])+", "
    content+=str(history.history["loss"][-1])+", "+str(history.history["val_loss"][-1])+", "
    content+=str(len(history.history["loss"]))+", "+str(f1)+", "+str(prec)+", "+str(rec)

    F.write(header)
    F.write(content)
    F.close()


