#! /usr/bin/env python
from setuptools import setup

setup(
    name='HMM',
    version='1.0.0',

    description='Implementation of the Linear Sparse Version Algorithms to Hidden Markov Model',
    long_description= 'This is an implementation of the memory sparse version of the Baum-welch and Viterbi algorithms to HMM.',
    url='https://github.com/xiaozhouw/663',
    author='Hao Sheng, Xiaozhou Wang',
    author_email='hao.sheng@duke.edu,xiaozhou.wang@duke.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Topic :: Software Development :: Build Tools',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    keywords='hidden markov model',
    install_requires=['numpy'],
)
