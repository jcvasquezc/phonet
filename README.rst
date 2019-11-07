======================
Phonet
======================

.. image:: https://readthedocs.org/projects/phonet/badge/?version=latest
:target: https://phonet.readthedocs.io/en/latest/?badge=latest
:alt: Documentation Status

.. image:: https://travis-ci.org/jcvasquezc/phonet.svg?branch=master
    :target: https://travis-ci.org/jcvasquezc/phonet

This toolkit compute posteriors probabilities of phonological classes from audio files for several groups of phonemes according to the mode and manner of articulation.

If you are not sure about what phonological classes are, have a look at this
`Phonological classes tutorial <http://research.cs.tamu.edu/prism/lectures/sp/l3.pdf>`_


`Project Documentation <http://phonet.readthedocs.org/en/latest/>`_

`Paper <ttp://dx.doi.org/10.21437/Interspeech.2019-1405>`_

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
      phon=Phonet([phonclass])
      get_phon_wav(self, audio_file, feat_file, plot_flag=True)

============= ===========
Parameters     Description
============= ===========
audio_file    file audio (.wav) sampled at 16 kHz
feat_file     file (.csv) to save the posteriors for the phonological classes
phonclass     list of phonological classes to be evaluated
              The list of phonological classes include:
              "consonantal", "back", "anterior", "open", "close", "nasal", "stop",
              "continuant",  "lateral", "flap", "trill", "voice", "strident",
              "labial", "dental", "velar", "pause", "vocalic" or "all"
plot_flag     True or False, whether you want plots of phonological classes or not
returns			  It crates the feat_file with the estimation of the phonological classes for each time-frame of the audio file.
============= ===========

Training
====================================

If you want to train Phonet in your own language, or specific phonological classes that are not defined here, please refer to the folder `train <https://github.com/jcvasquezc/phonet/tree/master/train>`_ and follow the instructions there.

If you experienced problems with the Training process, please send me an email `<juan.vasquez@fau.de>`


Reference
==================================

Phonet is available for research purposes

If you use Phonet, please cite the following paper.

@inproceedings{Vasquez-Correa2019,
  author={J. C. Vásquez-Correa and P. Klumpp and J. R. Orozco-Arroyave and E. N\"oth},
  title={{Phonet: A Tool Based on Gated Recurrent Neural Networks to Extract Phonological Posteriors from Speech}},
  year=2019,
  booktitle={Proc. Interspeech 2019},
  pages={549--553},
  doi={10.21437/Interspeech.2019-1405},
  url={http://dx.doi.org/10.21437/Interspeech.2019-1405}
}
