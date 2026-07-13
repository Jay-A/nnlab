# nnlab

## Neural Network Lab for Experimental Learning Systems

[Documentation](https://jay-a.github.io/nnlab/) | [Source Code](https://github.com/Jay-A/nnlab)

---

## Overview

So many conversations about Machine Learning (ML), and Neural Networks (NN) in
general, quickly become focused on the idea of the "AI black box" — systems
that appear almost magical because of the complexity hidden inside them.

That perspective misses something important: while we may not be able to
analytically describe every internal behavior of a modern neural network, we
can still study how these systems work by understanding and isolating their
individual components.

`nnlab` is built around the idea that neural networks are not mysterious
monolithic systems, but collections of interacting parts. By separating and
experimenting with those parts, we can observe how small changes in one
component influence the behavior of the complete learning system.

The goal of this project is not to compete with large-scale machine learning
frameworks or optimize only for benchmark performance. Instead, `nnlab`
provides a computational lab.oratory for exploring the fundamentals of neural
network behavior through controlled experiments, configurable components, and
visual analysis.

The framework focuses on studying the relationships between:

- model architecture,
- activation function behavior,
- optimization strategies,
- data representation,
- feature extraction,
- and learning outcomes.

By making these components easier to isolate and modify, `nnlab` aims to make
neural network experimentation more transparent, reproducible, and accessible.

The project begins with fundamental feed-forward neural networks and extends
toward broader learning systems including classification problems, sensor
applications, audio analysis, and future convolutional and recurrent models.

---

## Motivation

Many machine learning frameworks are designed around a workflow like:

```text
define model → train model → evaluate performance
```

This is an extremely effective approach for building useful systems, but it
can make it difficult to explore the underlying behavior of the individual
components that make those systems work.

`nnlab` takes a more experimental approach:

```text
define component → modify behavior → observe effects → understand interactions
```

The purpose is to create a space where neural networks can be studied as
systems of interacting mathematical components.

Questions that motivate the project include:

- How does the shape of an activation function influence learning behavior?
- How do different parameterizations affect optimization?
- How do architectural choices change representation and generalization?
- How do optimization strategies interact with the structure of a model?
- How can learned representations be applied to real sensor and signal data?

Rather than treating neural networks only as tools for producing predictions,
`nnlab` treats them as objects of study.

The goal is to provide a framework where neural networks can be explored
through experiments that are small enough to understand, controlled enough to
reproduce, and flexible enough to reveal how the pieces of a learning system
influence one another.

---

## Project Goals

`nnlab` is designed around the idea of a **computational lab.oratory** for
neural networks.

Key goals:

- Explore neural network behavior through controlled experiments.
- Compare different activation function families.
- Study the influence of activation shape and parameterization on learning.
- Provide clear visualizations of training dynamics.
- Support reproducible computational experiments.
- Maintain a modular architecture suitable for future extensions.
- Provide a foundation for experimenting with real-world signal and sensor
  applications.

---

## Current Architecture

`nnlab` is organized around independent components that can be combined into
complete learning systems.

Current components include:

### Layers

Neural network computation is represented through modular layers.

Current implementations include:

- Fully connected dense layers.
- Activation layers.
- Forward propagation.
- Backward propagation.
- Parameter and gradient tracking.

Layers expose a consistent interface:

```text
input
  ↓
forward()
  ↓
output

gradient
  ↓
backward()
  ↓
previous layer gradient
```

This allows models to be constructed as reusable computational graphs.

---

### Models

Current model support includes feed-forward networks.

Example:

```text
input
 ↓
dense layer
 ↓
activation
 ↓
dense layer
 ↓
prediction
```

Models coordinate:

- forward evaluation,
- backward propagation,
- parameter collection.

---

### Optimizers

`nnlab` includes optimization components responsible for updating trainable
parameters.

Current optimizer support includes:

- Stochastic Gradient Descent (SGD)

The separation between gradients and optimization allows experiments with
different learning strategies without modifying the underlying model.

---

### Training

Training utilities provide the connection between:

```text
data
 ↓
model
 ↓
loss
 ↓
backward pass
 ↓
optimizer
```

This supports repeatable experiments involving:

- epochs,
- loss tracking,
- parameter updates,
- training history visualization.

---

## Transition Kernel Framework

A central idea of `nnlab` is the use of parameterized transition kernels as
activation function building blocks.

Rather than treating activation functions as isolated formulas, `nnlab`
separates the mathematical shape of an activation from its configurable
parameters.

A parameterized activation is defined as:

$$
\phi(x)=aK\left(\frac{x-c}{s}\right)+b
$$

where:

- \(K\) is a transition kernel defining the underlying mathematical shape,
- \(c\) controls the activation center,
- \(s\) controls the transition scale,
- \(a\) controls output scaling,
- \(b\) controls output offset.

This allows activation behavior to be adjusted continuously without creating
separate activation classes for every variation.

Current transition kernel families include:

- **Logistic**  
  Smooth bounded sigmoid transition. Useful as a general-purpose activation
  shape and a baseline for comparison.

- **Arctangent**  
  Smooth sigmoid-like transition with heavier tails than logistic behavior.

- **Gaussian CDF**  
  Smooth cumulative transition derived from the normal distribution.

- **Gaussian RBF**  
  Localized radial response useful for basis-function style activations and
  localized feature responses.

- **Piecewise Linear**  
  Simple controllable transition useful for approximating ReLU-like behavior.

- **Polynomial**  
  Flexible algebraic transition families for exploring non-sigmoidal behavior.

This framework enables systematic comparison of activation shapes, parameter
effects, and learning dynamics.

---

## Feature Representation

Modern learning systems often depend on transforming raw observations into
useful numerical representations.

`nnlab` is beginning to explore feature extraction as a separate component from
the neural network itself.

The intended workflow is:

```text
raw signal
    ↓
feature extraction
    ↓
neural network
    ↓
prediction
```

Potential applications include:

- sensor telemetry,
- audio signals,
- industrial monitoring,
- autonomous systems.

This separation keeps feature engineering independent from model architecture
and allows different representations to be compared experimentally.

---

## Experimental Applications

Current experiments explore the progression:

```text
single neuron
      ↓
feed-forward networks
      ↓
activation kernel comparison
      ↓
architecture exploration
      ↓
optimization studies
      ↓
sensor-based applications
```

Example applications include:

### Drone Sensor Health Monitoring

Exploring how neural networks can learn relationships between telemetry signals
such as:

- vibration,
- temperature,
- battery state,
- system measurements.

### Audio Event Recognition

Exploring how signal representations can be transformed into features suitable
for learning tasks such as:

- machine sound analysis,
- anomaly detection,
- event classification.

---

## Planned Features

### Models

- Feed-forward neural networks
- Classification models
- Convolutional neural networks
- Recurrent architectures
- Additional experimental architectures

### Problems

- Function regression
- Classification
- Signal analysis
- Sensor-based modeling
- Scientific computing applications

### Experiments

- Activation function comparisons
- Architecture studies
- Optimization experiments
- Learning curve analysis
- Parameter sensitivity studies
- Representation comparisons

### Visualization

Planned visualizations include:

- Loss curves
- Learned functions
- Error surfaces
- Activation function shapes
- Network architecture diagrams
- Experiment comparisons
- Sensor and signal analysis plots

---

## Documentation

Documentation will be hosted through GitHub Pages:

[Documentation](https://jay-a.github.io/nnlab/)

The documentation will include:

- Getting started guides,
- API documentation,
- Mathematical background,
- Experiment descriptions,
- Automatically generated figures and examples.

Documentation will be built using Sphinx, with support for automated
generation of experimental results and visualizations.

---

## Installation

`nnlab` is developed using an isolated Python virtual environment to provide
reproducible dependencies. The core library does not require Jupyter; notebooks
are an optional environment for interactive experiments and visualization.

The intended workflow is:

```bash
git clone https://github.com/Jay-A/nnlab.git
cd nnlab

py -3.12 -m venv .venv
```

Activate the virtual environment:

**Windows**

```cmd
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Upgrade packaging tools:

```bash
python -m pip install --upgrade pip
```

Install the development environment:

```bash
pip install -e ".[dev,docs,notebooks]"
```

This installs:

- core scientific computing dependencies,
- development tools,
- documentation tools,
- Jupyter notebook support.

To make the environment available as a Jupyter kernel:

```bash
python -m ipykernel install --user --name nnlab --display-name "Python (nnlab)"
```

Jupyter notebooks can now be launched using the `Python (nnlab)` kernel.

For example:

```bash
jupyter lab
```

or:

```bash
jupyter notebook
```

Examples and scripts can also be executed directly from the terminal without
Jupyter.