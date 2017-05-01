# Sta-663 Final Project

## Project description

This project implements the memory sparse version of Viterbi algorithm and Baum-Welch algorithm to hidden Markov Model. 

The whole project is based on the paper ''Implementing EM and Viterbi algorithms for Hidden Markov Model in linear memory'', written by Alexander Churbanov and Stephen Winters-Hilt.

## Authors
Hao Sheng, Xiaozhou Wang
{hao.sheng,xiaozhou.wang}@duke.edu

## Install instructions

Please run following commands in bash if you have not cloned our GitHub repository:

```
$ git clone https://github.com/xiaozhouw/663.git
$ cd 663
$ python3 setup.py install
```

If you have already cloned our GitHub repository, please change your working directory to 663 (our repo) and run following command in bash:

```
$ python3 setup.py install
```
## Files descriptions

### `Code_Report.ipynb`

This file contains the code report part of our reports. This iPython Notebook includes code and partial explanation of:
- Benchmarking
- Simulations
- Applications
- Comparative Analysis

### `Setup.py` and `HMM` folder

`setup.py` is the script for installing our HMM package. 
In `HMM` folder, there are 3 scripts: `__init__.py`, `hmm.py` and `hmm_unoptimized.py`.

- `__init__.py`: give specification of how this package will be loaded.
- `hmm.py`: actual script for functionality.
- `hmm_unoptimized.py`: (partially) unoptimized script. Used in Benchmarking.

### `data` folder
The data we used in application (`weather-test2-1000.txt`) is stored in `data` folder. The data is retrieved from [HMM Programming Project](https://inst.eecs.berkeley.edu/~cs188/sp08/projects/hmm/project_hmm.html). How this synthesized data is generated can be found on [HMM Tutorial](https://inst.eecs.berkeley.edu/~cs188/sp08/slides/tr-98-041-1.pdf).

### `report.pdf`

This file contains the written report for our project.
