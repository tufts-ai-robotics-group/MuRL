.. MuRL documentation master file, created by
   sphinx-quickstart on Mon Aug 19 11:47:31 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MuRL documentation
==================

MuLIP-RL (MuRL) is a reinforcement learning framework focused on single file implementations of RL algorithms. MuRL is inspired by other implementations like tianshou and rllib. Like these other implementations, MuRL makes use of parallelism to speed up experiment's. MuRL provides a set of abstract base classes for both policy gradient and action-value methods to enable new algorithms to be added and supported easily. Unlike more complex approaches, these ABC's enforce as little as possible to allow flexibility between algorithm implementations. This is why MuRL is able to conduct experiment's with both tabular and estimation approaches.

This documentation will go through the process of setting up MuRL and using features like the config parser and callback mechanisms.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   config
   callbacks
