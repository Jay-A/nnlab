Activation Kernels
==================


nnlab explores parameterized activation functions built from
transition kernels.


A parameterized activation:

.. math::

   \phi(x)=aK((x-c)/s)+b


where:

- K defines the transition shape,
- c controls the center,
- s controls the scale,
- a controls amplitude,
- b controls offset.