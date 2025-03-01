import os
import glob
import shutil

from setuptools import setup, find_namespace_packages

setup(name                 = "oarfish",
      version              = "0.1.0",
      description          = "LWATV image classifier",
      long_description     = "Deep learning for classifying LWATV images",
      author               = "J. Dowell",
      author_email         = "jdowell@unm.edu",
      license              = 'BSD3',
      packages             = find_namespace_packages(),
      scripts              = glob.glob('scripts/*.py'),
      python_requires      = '>=3.8',
      install_requires     = ['numpy', 'zmq'],
      zip_safe             = False
)
