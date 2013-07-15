Python bindings for AD3
=======================

Author: Andreas Mueller <amueller@ais.uni-bonn.de>

Build Instructions
------------------
The Python bindings require Cython.
To build the Python bindings, 

```bash
python setup.py install
```

to install the bindings systemwide

or


```bash
python setup.py build_ext -i
```

to install them in AD3/python/ad3 directory


See ``example_grid.py`` or the notebook for an example.
