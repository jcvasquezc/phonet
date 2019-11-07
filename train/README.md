#Training phonet

To train phonet with a different dataset in a different languages, follow the next steps.

1. Install dependencies 

```pip install -r requirements.txt```

2. Feature extraction

Extract the Mel-filterbank energies from the training and test sets, using the script ```extract_feat.py``` as follows

```python extract_feat.py <path_audios> <path_to_save_features>```

Example with the CIEMPIESS database

```python extract_feat.py ../train_data/audio_same/ ../features/train/```
```python extract_feat.py ../test_data/audio_same/ ../features/test/```


3. Transform the phonological classes you want to train. Edit the file ```Phonological.py``` according to your classes. 

Edit variable ```list_phonological``` according to the phonological classes to train

Phonemes should be written in XAMPA symbols


4. Read and process the textgrid labels ```read_textgrids.py``` as follows

The ```textgrid``` files should be ```PRAAT``` labeled files with Xampa symbols.

You can obtain them using the [WebMAUS forced alignemt tool](https://clarin.phonetik.uni-muenchen.de/BASWebServices/interface/WebMAUSBasic)

Once you have the textgrid, you can use the following script.

```python read_textgrids.py <path_textgrids> <path_labels>```


Example with the CIEMPIESS database

```python read_textgrids.py ../train_data/textgrid/ ../labels/train/```
```python read_textgrids.py ../test_data/textgrid/ ../labels/test/```

5. Get the feature matrices for train and validation to train phonet with ```get_matrices_labels.py``` as follows

```python get_matrices_labels.py <path_features_in> <path_labels_in> <path_sequences_out>```

From the Example

```python get_matrices_labels.py ../features/train/ ../labels/train/ ../seq_train/```
```python get_matrices_labels.py ../features/test/ ../labels/test/ ../seq_test/```



6. Train a model for each phonological class using  ```main_train_RNN.py``` or a new model using a multi task learning strategy
using ```main_train_RNN_MT.py``` as follows

For the bank of parallel RNNs as in the paper an individual network has o be trained for each phonological class

```python main_train_RNN.py <path_seq_train>  <path_seq_test_test> <path_results> <phonological_class>```

Example 

```python main_train_RNN.py ../seq_train/ ../seq_test/ ../results/test_stop/ stop```
```python main_train_RNN.py ../seq_train/ ../seq_test/ ../results/test_nasal/ nasal```


7. In addition, a new model can be trained using a multi-task learning strategy. Instead of the bank of parallel RNNs, a single neural netowrk is trained.
The results a re similar to the obtained in the original paper. However, this version converge faster.

```python main_train_RNN_MT.py <path_seq_train>  <path_seq_test_test> <path_results>```

Example

```python main_train_RNN_MT.py ../seq_train/ ../seq_test/ ../results/MT_test/```

The inference part of Phonet have to be adapted to the neural networks model trained with the multi-task learning strategy.
