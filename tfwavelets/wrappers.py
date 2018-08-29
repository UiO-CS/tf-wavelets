import numpy as np
import tfwavelets as tfw
import tensorflow as tf


def dwt1d(signal, wavelet="haar", levels=1):
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
    signal = signal.astype(np.float32)
    signal = np.expand_dims(signal, 0)
    signal = np.expand_dims(signal, -1)

    # Placeholder for input signal
    tf_signal = tf.placeholder(dtype=tf.float32, shape=[None, None, None])

    # Get filter coeffs from database
    filter_coeffs = _parse_wavelet(wavelet)

    # Set up tf graph
    output = tfw.nodes.dwt1d(tf_signal, filter_coeffs, levels)

    # Compute graph
    with tf.Session() as sess:
        signal = sess.run(output, feed_dict={tf_signal: signal})

    # Remove empty dimensions and return
    return np.squeeze(signal)


def dwt2d(signal, wavelet="haar", levels=1):
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
    signal = signal.astype(np.float32)
    signal = np.expand_dims(signal, -1)

    # Placeholder for input signal
    tf_signal = tf.placeholder(dtype=tf.float32, shape=signal.shape)

    # Get filter coeffs from database
    filter_coeffs = _parse_wavelet(wavelet)

    # Set up tf graph
    output = tfw.nodes.dwt2d(tf_signal, filter_coeffs, levels)

    # Compute graph
    with tf.Session() as sess:
        signal = sess.run(output, feed_dict={tf_signal: signal})

    # Remove empty dimensions and return
    return np.squeeze(signal)


def _parse_wavelet(wavelet_name):
    """
    Look for wavelet coeffs in database, and return them if they exists

    Args:
        wavelet_name (str):     Name of wavelet

    Returns:
        (np.ndarray, np.ndarray): Filters for wavelet

    Raises:
        ValueError: If wavelet is not supported
    """
    if wavelet_name == "haar":
        return tfw.dwtcoeffs.haar
    elif wavelet_name == "db2":
        return tfw.dwtcoeffs.db2
    else:
        raise ValueError("dwt1d does not support wavelet {}".format(wavelet))
