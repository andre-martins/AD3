
# AD3+

This project is a fork from André Martins's project [AD3](https://github.com/andre-martins/AD3).

We extend here the __Python API to AD3__ to support:
- hard-logic constraints in inference methods
- inference on graph where nodes have different natures

We did those extensions in order to extend the pystruct structured learning library. See [Pystruct+](https://github.com/jlmeunier/pystruct)

100% ascendant compatible, so your code should work on AD3+, if it worked with AD3.


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


You can contact the author on [github](https://github.com/jlmeunier/AD3). Comments and contributions are welcome.

## EU Grant
Copyright Xerox(C) 2017 JL. Meunier

Developed for the EU project READ. The READ project has received funding 
from the European Union's Horizon 2020 research and innovation programme 
under grant agreement No 674943.

## Installation
See [AD3](https://github.com/andre-martins/AD3) documentation and/or [pystruct+](https://github.com/jlmeunier/pystruct)


## Change Log

### Release 2.2.1

- same code based on most recent version of AD3 "2.2.dev0" by vene 

### Release 2.1.2 

**Feb. 15th 2017**

- support graph of mutiple node natures
- support hard logic constraints
- support the mixture of both above

# AD3 Licence

AD3 (approximate MAP decoder with Alternating Direction Dual Decomposition)
Copyright (C) 2012
Andre Martins
Priberam Labs, Lisbon, Portugal &
Instituto de Telecomunicacoes, Instituto Superior Tecnico, Lisbon, Portugal
All Rights Reserved.

http://www.ark.cs.cmu.edu/AD3

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
