#!/usr/bin/env python

from setuptools import setup

setup(name='wsdt',
      version='0.1',
      description='Implementation of a distance-transform-based watershed algorithm',
      author='Timo Prange',
      author_email='timo.prange@iwr.uni-heidelberg.de',
      url='github.com/ilastik/wsdt',
      packages=['wsdt'],
      zip_safe=False,
      include_package_data=False
      #install_requires=['vigra']
     )
