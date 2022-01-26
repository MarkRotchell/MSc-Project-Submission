.. pyGPB documentation master file, created by
   sphinx-quickstart on Mon Aug 23 15:09:33 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyGPB's documentation!
=================================

How do I install it?
--------------------

Download the `base-pyGPB` folder, navigate to it in your terminal and run::

   pip install dist/pyGPB-0.0.1.tar.gz

How do I use pyGPB?
-------------------

The following example shows how to generate a GPB object representing a GPB random
variable and then print its probability mass vector::

   from pyGPB import GPB

   x = GPB(probs=[0.1,0.2,0.3], weights=[1,2,3])
   print(x.pmf_vec)

This will print the following::

   [0.504 0.056 0.126 0.23  0.024 0.054 0.006]

Table Of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   self
   GPB
   LFGPB

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


