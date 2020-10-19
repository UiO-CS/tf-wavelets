"""
The 'wrappers' module contains methods that wraps around the functionality in nodes. The
construct a full TF graph, launches a session, and evaluates the graph. Intended to be
used when you just want to compute the DWT/IDWT of a signal.
"""

import numpy as np
import tfwavelets as tfw
import tensorflow as tf


def dwt1d(signal, wavelet="haar", levels=1, npdtype=np.float32):
    """
    Computes the DWT of a 1D signal.

    Args:
        signal (np.ndarray):    A 1D array to compute DWT of.
        wavelet (str):          Type of wavelet (haar or db2)
        levels (int):           Number of DWT levels

    Returns:
        np.ndarray: The DWT of the signal.

    Raises:
        ValueError: If wavelet is not supported
    """
    # Prepare signal for tf. Turn into 32bit floats for GPU computation, and
    # expand dims to make it into a 3d tensor so tf.nn.conv1d is happy
    signal = signal.astype(npdtype)
    signal = np.expand_dims(signal, 0)
    signal = np.expand_dims(signal, -1)

    # Construct and compute TF graph
    return _construct_and_compute_graph(
        signal,
        tfw.nodes.dwt1d,
        _parse_wavelet(wavelet),
        levels
    )


def dwt2d(signal, wavelet="haar", levels=1, npdtype=np.float32):
    """
    Computes the DWT of a 2D signal.

    Args:
        signal (np.ndarray):    A 2D array to compute DWT of.
        wavelet (str):          Type of wavelet (haar or db2)
        levels (int):           Number of DWT levels

    Returns:
        np.ndarray: The DWT of the signal.

    Raises:
        ValueError: If wavelet is not supported
    """
    # Prepare signal for tf. Turn into 32bit floats for GPU computation, and
    # expand dims to make it into a 3d tensor so tf.nn.conv1d is happy
    signal = signal.astype(npdtype)
    signal = np.expand_dims(signal, -1)

    # Construct and compute TF graph
    return _construct_and_compute_graph(
        signal,
        tfw.nodes.dwt2d,
        _parse_wavelet(wavelet),
        levels
    )


def idwt1d(signal, wavelet="haar", levels=1, npdtype=np.float32):
    """
    Computes the inverse DWT of a 1D signal.

    Args:
        signal (np.ndarray):    A 1D array to compute IDWT of.
        wavelet (str):          Type of wavelet (haar or db2)
        levels (int):           Number of DWT levels

    Returns:
        np.ndarray: The IDWT of the signal.

    Raises:
        ValueError: If wavelet is not supported
    """
    # Prepare signal for tf. Turn into 32bit floats for GPU computation, and
    # expand dims to make it into a 3d tensor so tf.nn.conv1d is happy
    signal = signal.astype(npdtype)
    signal = np.expand_dims(signal, 0)
    signal = np.expand_dims(signal, -1)

    # Construct and compute TF graph
    return _construct_and_compute_graph(
        signal,
        tfw.nodes.idwt1d,
        _parse_wavelet(wavelet),
        levels
    )


def idwt2d(signal, wavelet="haar", levels=1, npdtype=np.float32):
    """
    Computes the inverse DWT of a 2D signal.

    Args:
        signal (np.ndarray):    A 2D array to compute iDWT of.
        wavelet (str):          Type of wavelet (haar or db2)
        levels (int):           Number of DWT levels

    Returns:
        np.ndarray: The IDWT of the signal.

    Raises:
        ValueError: If wavelet is not supported
    """
    # Prepare signal for tf. Turn into 32bit floats for GPU computation, and
    # expand dims to make it into a 3d tensor so tf.nn.conv1d is happy
    signal = signal.astype(npdtype)
    signal = np.expand_dims(signal, -1)

    # Construct and compute TF graph
    return _construct_and_compute_graph(
        signal,
        tfw.nodes.idwt2d,
        _parse_wavelet(wavelet),
        levels
    )


def _construct_and_compute_graph(input_signal, node, wavelet_obj, levels, npdtype):
    """
    Constructs a TF graph processing the input signal with given node and evaluates it.

    Args:
        input_signal:       Input signal. A 3D array with [batch, signal, channels]
        node:               Node to process signal with, any kind of dwt/idwt
        wavelet_obj:        Wavelet object to pass to node
        levels:             Num of levels (passed to node)

    Returns:

    """
    if npdtype == np.float32:
        dtype=tf.float32;
    elif npdtype == np.float64:
        dtype=tf.float64;

    # Placeholder for input signal
    tf_signal = tf.placeholder(dtype=dtype, shape=input_signal.shape)

    # Set up tf graph
    output = node(tf_signal, wavelet=wavelet_obj, levels=levels)

    # Compute graph
    with tf.Session() as sess:
        signal = sess.run(output, feed_dict={tf_signal: input_signal})

    # Remove empty dimensions and return
    return np.squeeze(signal)


def _parse_wavelet(wavelet):
    """
    Look for wavelet coeffs in database, and return them if they exists

    Args:
        wavelet (str):     Name of wavelet

    Returns:
        (np.ndarray, np.ndarray): Filters for wavelet

    Raises:
        ValueError: If wavelet is not supported
    """
    if wavelet == "haar":
        return tfw.dwtcoeffs.haar
    elif wavelet == "db2":
        return tfw.dwtcoeffs.db2
    elif wavelet == "db3":
        return tfw.dwtcoeffs.db3
    elif wavelet == "db4":
        return tfw.dwtcoeffs.db4
    else:
        raise ValueError("dwt1d does not support wavelet {}".format(wavelet))
