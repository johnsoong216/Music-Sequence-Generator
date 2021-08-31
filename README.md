# Music-Sequence-Generation
CSC412 Winter 2020 Final Project

### Introduction
This project aims to generate polyphonic melodies with a linguistic approach, comparing performance between variants of hidden Markov models (HMM) and an Encoder-Decoder network that uses long-short term memory (LSTM) cells. The main objective of this project is to construct pleasant melodies that sound indistinguishable from human-composed ones. The project evaluates the models' performance by both quantitative and qualitative measures and discusses possible areas for exploration.


### How to run Encoder-Decoder Network
```shell
python run_lstm.py -i data_dir
```
See python file for additional options for input. 


### How to run HMM
Check HMM notebook for details


### Samples
HMM outputs are located in HMM folder, generated using furelise by Beethoven. <br>
LSTM outputs are located in LSTM folder, generated using compositions from beeth folder
