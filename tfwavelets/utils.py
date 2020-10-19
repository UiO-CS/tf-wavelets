"""
The 'utils' module contains some useful helper functions, mostly used during the
implementation of the other modules.
"""

import numpy as np
import tensorflow as tf


def adapt_filter(filter):
    """
    Expands dimensions of a 1d vector to match the required tensor dimensions in a TF
    graph.

    Args:
        filter (np.ndarray):     A 1D vector containing filter coefficients

    Returns:
        np.ndarray: A 3D vector with two empty dimensions as dim 2 and 3.

    """
    # Add empty dimensions for batch size and channel num
    return np.expand_dims(np.expand_dims(filter, -1), -1)


def to_tf_mat(matrices, dtype=tf.float32):
    """
    Expands dimensions of 2D matrices to match the required tensor dimensions in a TF
    graph, and wrapping them as TF constants.

    Args:
        matrices (iterable):    A list (or tuple) of 2D numpy arrays.

    Returns:
        iterable: A list of all the matrices converted to 3D TF tensors.
    """
    result = []

    for matrix in matrices:
        result.append(
            tf.constant(np.expand_dims(matrix, 0), dtype=dtype)
        )

    return result
