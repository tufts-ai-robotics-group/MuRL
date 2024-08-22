# MuRL
## Introduction

MuLIP-RL (MuRL) is a fledgling implementation of an RL library. It borrows key features from existing libraries like tianshou and rllib (namely the idea of rllib's complex callback system). This library has two main advantages over a more complex implementation:

- Makes use of single file implementations of algorithms
- Supports BOTH tabular and estimation algorithms

The default runner has the ability to run experiment's, described by json files, in parallel. A built in logging framework is used to save training and validation results, as well as provides a mechanism to monitor progress. 

Additional information on usage is provided in the autodocs.

## TODO

This library still has a number of missing features including:

- Lacking support for tensorboard or another system to make monitoring training progress
- A clearer mechanism to allow backing up and models during training and another to allow training be resumed after a system failure.
