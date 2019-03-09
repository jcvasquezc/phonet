======================
Phonet
======================

This toolkit compute posteriors probabilities of phonological classes from audio files for several groups of phonemes according to the mode and manner of articulation.

If you are not sure about what phonological classes are, have a look at this
`Phonological classes tutorial <http://research.cs.tamu.edu/prism/lectures/sp/l3.pdf>`_


`Project Documentation <http://phonet.readthedocs.org/en/latest/>`_

The list of the phonological classes available and the phonemes that are activated for each phonological class are observed in the following Table


The list of the phonological classes available and the phonemes that are activated for each phonological class are observed in the following Table


==================    ================================================================================
Phonological class    Phonemes
==================    ================================================================================
vocalic               /a/, /e/, /i/, /o/, /u/
consonantal           /b/, /tS/, /d/, /f/, /g/, /x/, /k/, /l/, /ʎ/, /m/, /n/, /p/, /ɾ/, /r/, /s/, /t/
back                  /a/, /o/, /u/
anterior              /e/, /i/
open                  /a/, /e/, /o/
close                 /i/, /u/
nasal                 /m/, /n/
stop                  /p/, /b/, /t/, /k/, /g/, /tS/, /d/
continuant            /f/, /b/, /tS/, /d/, /s/, /g/, /ʎ/, /x/
lateral               /l/
flap                  /ɾ/
trill                 /r/
voiced                /a/, /e/, /i/, /o/, /u/, /b/, /d/, /l/, /m/, /n/, /r/, /g/, /ʎ/
strident              /f/, /s/, /tS/
labial                /m/, /p/, /b/, /f/
dental                /t/, /d/
velar                 /k/, /g/, /x/
pause                 /sil/
==================    ================================================================================



Installation
============


From this repository::

    git clone https://github.com/jcvasquezc/phonet
    cd phonet
    python setup.py

Usage
=====

Supported features:

- Estimate probabilities of phonological classes for an audio file


`Example use <example.py>`_



 
Estimation of phonological classes
====================================


Estimate the phonological classes using the BGRU models for an audio file or for a folder that contains audio files inside::


      python
      def getphonclass(audio_file, feat_file, phonclass="all", plot_flag=True)

============= ===========
Parameter     Description
============= ===========
audio_file    file audio (.wav) or path with audio files inside, sampled at 16 kHz
feat_file     file (.csv) to save the probabilities for the phonological classes,
              or a folder to save the posteriors for different files when audio_file is a folder
phonclass     phonological class to be evaluated
              The list of phonological classes include:
              "consonantal", "back", "anterior", "open", "close", "nasal", "stop",
              "continuant",  "lateral", "flap", "trill", "voice", "strident",
              "labial", "dental", "velar", "pause", "vocalic" or "all"
plot_flag     True or False, whether you want plots of phonological classes or not
returns			  It crates the feat_file with the estimation of the phonological classes for each time-frame of the audio file.
============= ===========

