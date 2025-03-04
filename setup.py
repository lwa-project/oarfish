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
      classifiers          = ['Development Status :: 4 - Beta',
                              'Intended Audience :: Science/Research',
                              'License :: OSI Approved :: BSD License',
                              'Topic :: Scientific/Engineering :: Astronomy'],
      packages             = find_namespace_packages(),
      scripts              = glob.glob('scripts/*.py'),
      include_package_data = True,
      python_requires      = '>=3.8',
      install_requires     = ['numpy', 'pyzmq', 'astropy'],
      extra_requires       = {'full': ['torch', 'torchvision', 'scipy', 'scikit-learn'],
                              'client': []},
      zip_safe             = False
)
