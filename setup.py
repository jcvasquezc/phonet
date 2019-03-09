from setuptools import setup
from setuptools import find_packages

install_requires = [
    'tensorflow',
    'Keras',
    'pandas',
    'python_speech_features'
]

setup(
      name='phonet',
      version='1.0.0',
      description='Compute phonological posteriors from speech signals using a deep learning scheme',
      author='J. C. Vasquez-Correa',
      author_email='juan.vasquez@fau.de',
      url='https://github.com/jcvasquezc/phonet',
      download_url='https://github.com/jcvasquezc/phonet',
      license='GNU GPL v2',
      install_requires=install_requires,
      packages=find_packages(),
      dependency_links=['git+git://github.com/jameslyons/python_speech_features']
)
