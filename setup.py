#!/usr/bin/env python

import os
from setuptools import find_packages

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


readme = open('README.rst').read()


requires = [] #during runtime
tests_require=[] #for testing

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

setup(
    name='mypackage',
    version='0.0.1',
    description='Set of classes for kannada mnist challenge',
    long_description=readme,
    author='Adwaye Rambojun',
    url='https://github.com/adwaye',
    #packages=find_packages(PACKAGE_PATH, "test"),
    package_dir={'mypackage': 'mypackage'},
    include_package_data=True,
    install_requires=requires,
    license='GPLv3',
    zip_safe=False,
    keywords='kannadaMnist',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        "Intended Audience :: Science/Research",
        'Intended Audience :: Developers',
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    tests_require=tests_require,
)
