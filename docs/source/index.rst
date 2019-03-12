.. phonet documentation master file, created by
   sphinx-quickstart on Sat Mar  9 04:39:38 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Phonet's documentation!
==================================

This toolkit compute posteriors probabilities of phonological classes from audio files for several groups of phonemes according to the mode and manner of articulation.

If you are not sure about what phonological classes are, have a look at this
`Phonological classes tutorial <http://research.cs.tamu.edu/prism/lectures/sp/l3.pdf>`_


The code for this project is available at https://github.com/jcvasquezc/phonet .

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


.. toctree::
   :maxdepth: 3

   help



Supported features:

- :py:meth:`phonet.getphonclass` - Estimate the phonological classes for an audio file or for a folder that contains audio files inside.


To estimate the `nasal` phonological class::

    from phonet import Phonet

    file_audio="./audios/sentence.wav"
    file_feat="./phonclasses/sentence_nasal"
    phon.get_phon_wav(file_audio, file_feat, "nasal", True)

getphonclass will save a .csv file in "file_feat" with the probability of the nasal phonological class for every time frame of the audio file.


Install
-------------------------------------

From the source file::

    git clone https://github.com/jcvasquezc/phonet
    cd phonet
    python setup.py install

Methods
-------------------------------------

.. automodule:: phonet

    .. autoclass:: Phonet
      :members:


Indices and tables
-------------------------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



Help
-------------------------------------
If you have trouble with Phonet, please write to Camilo Vasquez at: juan.vasquez@fau.de
