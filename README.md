# NeuroRNNTools
PyTorch models for RNNs commonly used in computational neuroscience.

PyTorch nn modules and associated utilities for recurrent rate network models and spiking network models commonly used in computational neuroscience. 

# Rate network models:

Creating a rate network module:
model = RateModel(recurrent, readin=None, readout=None, f='tanh', eta=1, rho_recurrent=1, rho_input=1, rho_output=1, bias_recurrent=False, bias_output=False, Network_Type='R')

