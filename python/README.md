Python bindings for AD3
=======================

Authors: 
    Andreas Mueller <amueller@ais.uni-bonn.de>
    Vlad Niculae <vlad@vene.ro>
    Jean-Luc Meunier <jean-luc.meunier@naverlabs.com>


Installation Instructions
-------------------------

The wrapper is available on PyPI. Wheels are distributed for most platforms.

```bash
pip install ad3
```


Build Instructions
------------------
The Python bindings require Cython.
To build the Python bindings use the following commands at the top level:

```bash
pip install .
```

to install the bindings systemwide

or


```bash
pip install -e .
```

to install them locally.


# Support for logic constraints and typed nodes

This section documents support for
- hard-logic constraints in inference methods
- inference on graph where nodes have different natures

We did those extensions in order to extend the pystruct structured learning
library. See [Pystruct+](https://github.com/jlmeunier/pystruct)

Extension originally implemented by JL Meunier, 2017.
Developed for the EU project READ. The READ project has received
funding from the European Union's Horizon 2020 research and innovation programme
under grant agreement No 674943.


## Hard Logic Constraints
As explained in André's ICML paper [1], one can **binarize the graph** and make inference on boolean values.
Exploiting this method, we support logical constraints when doing inference.

[1] André F. T. Martins, M�rio A. T. Figueiredo, Pedro M. Q. Aguiar, Noah A. Smith, and Eric P. Xing.
"An Augmented Lagrangian Approach to Constrained MAP Inference."
International Conference on Machine Learning (ICML'11), Bellevue, Washington, USA, June 2011.

See also 
[2] Jean-Luc Meunier, "Joint Structured Learning and Predictions under Logical Constraints in Conditional Random Fields"
Conference CAp 2017
 arXiv:1708.07644

## Nodes of Different Nature
When the nodes of the graph are of different nature, their number of possible states may differ from each other. Provided the definition of the number of states per **type of node** , the inference method deals gracefully with this situation.

## Hard Logic and Node of Multiple Nature
Yes, the combination of both is possible and works fine! :-)

