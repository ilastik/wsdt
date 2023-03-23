#!/usr/bin/env python

from setuptools import setup
import setuptools_scm

_version = setuptools_scm.get_version(write_to="wsdt/_version.py")

setup(name='wsdt',
      version=_version,
      description='Implementation of a distance-transform-based watershed algorithm',
      author='Timo Prange',
      author_email='timo.prange@iwr.uni-heidelberg.de',
      url='github.com/ilastik/wsdt',
      packages=['wsdt'],
      zip_safe=False,
      include_package_data=False
      #install_requires=['vigra']
     )
