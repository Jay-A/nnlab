# nnlab

## Neural Network Laboratory for Experimental Learning Systems

---

## Overview

`nnlab` is a modular framework for experimenting with neural network architectures, activation functions, optimization strategies, and learning dynamics.

The goal of the project is to provide an environment for studying how neural networks learn through reproducible experiments, interactive visualizations, and configurable computational models.

Rather than focusing only on training performance, `nnlab` emphasizes understanding the relationship between:

- model architecture,
- activation function behavior,
- optimization strategies,
- data representation,
- and learning outcomes.

The initial focus is on function approximation and regression problems, with future extensions toward classification, convolutional models, recurrent architectures, and other neural network approaches.

---

## Project Goals

`nnlab` is designed around the idea of a **computational laboratory** for neural networks.

Key goals:

- Explore neural network behavior through controlled experiments.
- Compare different activation function families.
- Study the influence of activation shape and parameterization on learning.
- Provide clear visualizations of training dynamics.
- Support reproducible computational experiments.
- Maintain a modular architecture suitable for future extensions.

---

## Transition Kernel Framework

A central idea of `nnlab` is the use of parameterized transition kernels as activation function building blocks.

Instead of treating activations as isolated functions, `nnlab` studies families of functions of the form:

\[
\phi(x)=a\,K\left(\frac{x-c}{w}\right)+b
\]

where:

- \(K\) is a transition kernel,
- \(c\) controls the activation center,
- \(w\) controls transition width,
- \(a\) controls output scaling,
- \(b\) controls output offset.

Initial kernel families include:

- Logistic
- Gaussian
- Cauchy
- Polynomial
- Piecewise Linear

This framework allows systematic comparison of activation shapes and their effects on learning behavior.

---

## Planned Features

### Models

- Feed-forward neural networks
- Convolutional neural networks
- Recurrent architectures
- Future experimental architectures

### Problems

- Function regression
- Classification
- Data-driven modeling problems
- Scientific computing applications

### Experiments

- Activation function comparisons
- Architecture studies
- Optimization experiments
- Learning curve analysis
- Parameter sensitivity studies

### Visualization

Planned visualizations include:

- Loss curves
- Learned functions
- Error surfaces
- Activation function shapes
- Network architecture diagrams
- Experiment comparisons

---

## Documentation

Documentation will be hosted through GitHub Pages:

[Documentation](https://jay-a.github.io/nnlab/)

The documentation will include:

- Getting started guides
- API documentation
- Mathematical background
- Experiment descriptions
- Automatically generated figures and examples

Documentation will be built using Sphinx, with support for automated generation of experimental results and visualizations.

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