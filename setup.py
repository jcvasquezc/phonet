from setuptools import setup
from setuptools import find_packages

install_requires = [
    'tensorflow',
    'Keras',
    'pandas',
    'pysptk',
    'six',
    'matplotlib',
    'python_speech_features',
    'tqdm',
    'scikit_learn',
    'setuptools',
    'numpy',
    'scipy',
    'matplotlib',
]

setup(
    name='phonet',
    version='0.3',
    description='Compute phonological posteriors from speech signals using a deep learning scheme',
    author='J. C. Vasquez-Correa',
    author_email='juan.vasquez@fau.de',
    url='https://github.com/jcvasquezc/phonet',
    download_url='https://github.com/jcvasquezc/phonet/archive/0.3.tar.gz',
    license='MIT',
    install_requires=install_requires,
    packages=find_packages(),
    keywords = ['phonological', 'speech', 'speech features', 'articulatory features', 'phoneme recognition'],
    dependency_links=['git+git://github.com/jameslyons/python_speech_features'],
    classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],

)




      





