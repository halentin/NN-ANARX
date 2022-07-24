# NN-ANARX
This repository contains a PyTorch-implementation of NN-ANARX. NN-ANARX is a class of nonlinear-system-models based on neural networks, that can be converted to a state-space-representation. This conversion was first described [here](https://www.sciencedirect.com/science/article/abs/pii/S000510980600118X). 
## Features
* Creation of neural network based NARX, -ANARX and SANARX models in SISO and MISO configuration
* Open-Loop-Training
* Closed-Loop-Prediction and Training (based on a variant of backpropagation-trough-time)
* Conversion of NN-ANARX-Models to state-space-representation
* Computation of optimal control input for SISO-NN-SANARX-models [here](https://ieeexplore.ieee.org/abstract/document/4433965)
* Export of all models (including the state-space-representation) as ONNX 

## Usage
Have a look at the Jupyter-Notebooks in the src-Folder! They contain lots of examples and explanation on how to work with this library.

## Origin
This code is part of the results of a project on data-based nonlinear system identification and control. As part of this project we did not only experiment with this NN-ANARX-based control approach, we also used Reinforcement-Learning for nonlinear-systems-control. More results from that project can be found [here](https://github.com/jubra97/Projektmodul). 

## Author
All of the code in this repository was written by myself. The WandB-script was adapted from an example script.
