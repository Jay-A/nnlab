Backpropagation
===============


Backpropagation computes gradients through the computational graph.


The process:

.. code-block::

   prediction
        |
        v
   loss gradient
        |
        v
   backward propagation
        |
        v
   parameter updates