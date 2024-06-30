# TKan 

TKan is a project implementing an easily extendable KAN layer for PyTorch. The library contains a couple of implementations for different KAN activation functions and a set of utilities to easily set up a KAN.

## Installation

Right now, there is no pip distribution of this library. However, you can directly install from this git repository:
```bash
pip install git+https://github.com/fepegar/tkan.git
```

Alternatively, you can first clone the repository and then install from the local copy:
```bash
git clone https://github.com/fepegar/tkan.git
cd tkan
pip install -e .
```

## Usage

The provided layers are plane pytorch layers, so you can just use them in any PyTorch model. Here is an example of a small KAN:
```python
import torch.nn as nn
import tkan.nn as tnn

kan = nn.Sequential(
    tnn.HermiteKan(5, 3, order=4),
    tnn.HermiteKan(3, 1, order=3),
)
```

We provide some of the methods proposed by the authors of the excellent paper [KAN: Kolmogorov–Arnold Networks](https://arxiv.org/html/2404.19756v1) like [regularization losses](tkan/training/losses.py) and [pruning](tkan/training/pruning.py).
Refer to the [Toy example notebook](notebooks/toy_example.ipynb) for more details.

We also provide visualizations of KAN networks (heavily inspired by the visualization in the paper) in the [plotting utility](tkan/plotting/plotter.py). Once again there is a [notebook](notebooks/plotting.ipynb) demonstrating the details.

While we have not implemented the B-Spline layer proposed in the paper, we implemented a couple of simpler layers. 
The library was constructed with ease of implementation of new layer types in mind. 
In the [custom layer notebook](notebooks/custom_layer.ipynb) we demonstrate how to do this.

## References and similar projects:

I wrote this repository to get a deeper understanding of the KAN paper [1]() by Ziming Lui et al. and therefore, the code is strongly related to [their original repository](https://github.com/KindXiaoming/pykan).

1. [KAN: Kolmogorov–Arnold Networks](https://arxiv.org/html/2404.19756v1) by Ziming Lui et al.
2. [Efficient Kan](https://github.com/Blealtan/efficient-kan)
3. [Trochkan](https://github.com/1ssb/torchkan)