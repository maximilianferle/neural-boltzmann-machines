# Implementation of Long Short-Term Memory Conditional Restricted Boltzmann Machine (LSTMCRBM).

Uses LSTMs to encode medical time series data and Neural Boltzmann Machines (NBMs) implementation from this [paper](https://arxiv.org/abs/2305.08337)
to generate synthetic output conditioned on it.

All parameters of a CRBM are their own neural network that is a function of the
conditional input. 

This flexibility allows arbitrary neural networks to represent the bias and
variance of the visible units and the weights between visible and hidden units.