"""
The tfwavelets package offers ways to achieve discrete wavelet transforms in tensorflow.

The package consists of the following modules:

    * 'nodes' contains methods to construct TF subgraphs computing the 1D or 2D DWT or
      IDWT. Intended to be used if you need a DWT in your own TF graph.
    * 'wrappers' contains methods that wraps around the functionality in nodes. The
      construct a full TF graph, launches a session, and evaluates the graph. Intended to
      be used when you just want to compute the DWT/IDWT of a signal.
    * 'dwtcoeffs' contains predefined wavelets, as well as the classes necessary to
      create more user-defined wavelets.
    * 'utils' contains some useful helper functions, mostly used during the implementation
      of the other modules.
"""
from . import nodes
from . import wrappers
from . import dwtcoeffs
from . import utils

