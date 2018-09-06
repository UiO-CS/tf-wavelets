import tensorflow as tf
import numpy as np
from tfwavelets.dwtcoeffs import edge_matrices


def _adapt_filter(filter):
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


def _to_tf_mat(matrices):
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
            tf.constant(np.expand_dims(matrix, 0), dtype=tf.float32)
        )

    return result


def cyclic_conv1d(input_node, kernel_node, tl_node, tr_node, bl_node, br_node):
    """Cyclic convolution

    Args:
        input_node:  Input signal (3-tensor [batch, width, in_channels])
        kernel_node: Filter (3-tensor[filter_width, in_channels, out_channels])
        tl_node:     Top-left part of cyclic matrix
        tr_node:     Top-right part of cyclic matrix
        bl_node:     Bottom-left part of cyclic matrix
        br_node:     Bottom-right part of cyclic matrix

    Returns:
        Tensor with the result of a periodic convolution
    """
    inner = tf.nn.conv1d(input_node, kernel_node[::-1], stride=1, padding='VALID')

    input_shape = tf.shape(input_node)
    tl_shape = tf.shape(tl_node)
    tr_shape = tf.shape(tr_node)
    bl_shape = tf.shape(bl_node)
    br_shape = tf.shape(br_node)

    # Slices of the input signal corresponding to the corners
    tl_slice = tf.slice(input_node,
                        [0, 0, 0],
                        [-1, tl_shape[2], -1])
    tr_slice = tf.slice(input_node,
                        [0, input_shape[1] - tr_shape[2], 0],
                        [-1, tr_shape[2], -1])
    bl_slice = tf.slice(input_node,
                        [0, 0, 0],
                        [-1, bl_shape[2], -1])
    br_slice = tf.slice(input_node,
                        [0, input_shape[1] - br_shape[2], 0],
                        [-1, br_shape[2], -1])

    # TODO: It just werks (It's the magic of the algorithm). i.e. Why do we have to transpose?
    tl = tl_node @ tf.transpose(tl_slice, perm=[2, 1, 0])
    tr = tr_node @ tf.transpose(tr_slice, perm=[2, 1, 0])
    bl = bl_node @ tf.transpose(bl_slice, perm=[2, 1, 0])
    br = br_node @ tf.transpose(br_slice, perm=[2, 1, 0])

    head = tf.transpose(tl + tr, perm=[2, 1, 0])
    tail = tf.transpose(bl + br, perm=[2, 1, 0])

    return tf.concat((head, inner, tail), axis=1)


def dwt1d(input_node, filter_coeffs, levels=1):
    """
    Constructs a TF computational graph computing the 1D DWT of an input signal.

    Args:
        input_node:     A 3D tensor containing the signal. The dimensions should be
                        [batch, signal, channels].
        filter_coeffs:  A tuple of two 1D numpy arrays containing wavelet filter
                        coefficients. First filter is LP, second is HP.
        levels:         Number of levels.

    Returns:
        The output node of the DWT graph.
    """
    # TODO: Check that level is a reasonable number
    # TODO: Check types

    coeffs = [None] * (levels + 1)

    last_level = input_node

    lp_adapted = _adapt_filter(filter_coeffs[0])
    lp_mat = _to_tf_mat(edge_matrices(filter_coeffs[0], 0))

    hp_adapted = _adapt_filter(filter_coeffs[1])
    hp_mat = _to_tf_mat(edge_matrices(filter_coeffs[1], 1))

    tf_lp = tf.constant(lp_adapted, dtype=tf.float32, shape=[len(filter_coeffs[0]), 1, 1])
    tf_hp = tf.constant(hp_adapted, dtype=tf.float32, shape=[len(filter_coeffs[1]), 1, 1])

    for level in range(levels):
        # TODO: Convert stride kwarg to tuple
        # TODO: Actual convolution, not correlation
        # TODO: Periodic extention, not zero-padding
        # lp_res = tf.nn.conv1d(last_level, tf_lp, stride=2, padding="SAME")
        # hp_res = tf.nn.conv1d(last_level, tf_hp, stride=2, padding="SAME")

        lp_res = cyclic_conv1d(last_level, tf_lp, *lp_mat)[:, ::2, :]
        hp_res = cyclic_conv1d(last_level, tf_hp, *hp_mat)[:, ::2, :]

        last_level = lp_res
        coeffs[levels - level] = hp_res

    coeffs[0] = last_level
    return tf.concat(coeffs, axis=1)


def dwt2d(input_node, filter_coeffs, levels=1):
    """
    Constructs a TF computational graph computing the 2D DWT of an input signal.

    Args:
        input_node:     A 3D tensor containing the signal. The dimensions should be
                        [rows, cols, channels].
        filter_coeffs:  A tuple of two 1D numpy arrays containing wavelet filter
                        coefficients. First filter is LP, second is HP.
        levels:         Number of levels.

    Returns:
        The output node of the DWT graph.
    """
    # TODO: Check that level is a reasonable number
    # TODO: Check types

    coeffs = [None] * levels

    last_level = input_node
    m, n = int(input_node.shape[0]), int(input_node.shape[1])

    for level in range(levels):
        local_m, local_n = m // (2 ** level), n // (2 ** level)

        first_pass = dwt1d(last_level, filter_coeffs, 1)
        second_pass = tf.transpose(
            dwt1d(
                tf.transpose(first_pass, perm=[1, 0, 2]),
                filter_coeffs,
                1
            ),
            perm=[1, 0, 2])

        last_level = tf.slice(second_pass, [0, 0, 0], [local_m // 2, local_n // 2, 1])
        coeffs[level] = [
            tf.slice(second_pass, [0, local_n // 2, 0], [local_m // 2, local_n // 2, 1]),
            tf.slice(second_pass, [local_m // 2, 0, 0], [local_m // 2, local_n // 2, 1]),
            tf.slice(second_pass, [local_m // 2, local_n // 2, 0], [local_m // 2, local_n // 2, 1])
        ]

    for level in range(levels - 1, -1, -1):
        upper_half = tf.concat([last_level, coeffs[level][0]], 0)
        lower_half = tf.concat([coeffs[level][1], coeffs[level][2]], 0)

        last_level = tf.concat([upper_half, lower_half], 1)

    return last_level
