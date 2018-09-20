"""
The 'nodes' module contains methods to construct TF subgraphs computing the 1D or 2D DWT
or IDWT. Intended to be used if you need a DWT in your own TF graph.
"""

import tensorflow as tf


def cyclic_conv1d(input_node, filter_):
    """
    Cyclic convolution

    Args:
        input_node:  Input signal (3-tensor [batch, width, in_channels])
        filter_:     Filter

    Returns:
        Tensor with the result of a periodic convolution
    """
    # Create shorthands for TF nodes
    kernel_node = filter_.coeffs
    tl_node, tr_node, bl_node, br_node = filter_.edge_matrices

    # Do inner convolution
    inner = tf.nn.conv1d(input_node, kernel_node[::-1], stride=1, padding='VALID')

    # Create shorthands for shapes
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


def cyclic_conv1d_alt(input_node, filter_):
    """
    Alternative cyclic convolution. Uses more memory than cyclic_conv1d.

    Args:
        input_node:         Input signal
        filter_ (Filter):   Filter object

    Returns:
        Tensor with the result of a periodic convolution.
    """
    kernel_node = filter_.coeffs

    N = int(input_node.shape[1])

    start = N - filter_.num_neg()
    end = filter_.num_pos() - 1

    # Perodically extend input signal
    input_new = tf.concat(
        (input_node[:, start:, :], input_node, input_node[:, 0:end, :]),
        axis=1
    )

    # Convolve with periodic extension
    result = tf.nn.conv1d(input_new, kernel_node[::-1], stride=1, padding="VALID")

    return result


def upsample(input_node, odd=False):
    """Upsamples. Doubles the length of the input, filling with zeros

    Args:
        input_node: 3-tensor [batch, spatial dim, channels] to be upsampled
        odd:        Bool, optional. If True, content of input_node will be
                    placed on the odd indeces of the output. Otherwise, the
                    content will be places on the even indeces. This is the
                    default behaviour.

    Returns:
        The upsampled output Tensor.
    """

    columns = []
    for col in tf.unstack(input_node, axis=1):
        columns.extend([col, tf.zeros_like(col)])

    if odd:
        # https://stackoverflow.com/questions/30097512/how-to-perform-a-pairwise-swap-of-a-list
        # TODO: Understand
        # Rounds down to even number
        l = len(columns) & -2
        columns[1:l:2], columns[:l:2] = columns[:l:2], columns[1:l:2]

    # TODO: Should we actually expand the dimension?
    return tf.expand_dims(tf.concat(columns, 1), -1)


def dwt1d(input_node, wavelet, levels=1):
    """
    Constructs a TF computational graph computing the 1D DWT of an input signal.

    Args:
        input_node:     A 3D tensor containing the signal. The dimensions should be
                        [batch, signal, channels].
        wavelet:        Wavelet object
        levels:         Number of levels.

    Returns:
        The output node of the DWT graph.
    """
    # TODO: Check that level is a reasonable number
    # TODO: Check types

    coeffs = [None] * (levels + 1)

    last_level = input_node

    for level in range(levels):
        lp_res = cyclic_conv1d_alt(last_level, wavelet.decomp_lp)[:, ::2, :]
        hp_res = cyclic_conv1d_alt(last_level, wavelet.decomp_hp)[:, 1::2, :]

        last_level = lp_res
        coeffs[levels - level] = hp_res

    coeffs[0] = last_level
    return tf.concat(coeffs, axis=1)


def dwt2d(input_node, wavelet, levels=1):
    """
    Constructs a TF computational graph computing the 2D DWT of an input signal.

    Args:
        input_node:     A 3D tensor containing the signal. The dimensions should be
                        [rows, cols, channels].
        wavelet:        Wavelet object.
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

        first_pass = dwt1d(last_level, wavelet, 1)
        second_pass = tf.transpose(
            dwt1d(
                tf.transpose(first_pass, perm=[1, 0, 2]),
                wavelet,
                1
            ),
            perm=[1, 0, 2]
        )

        last_level = tf.slice(second_pass, [0, 0, 0], [local_m // 2, local_n // 2, 1])
        coeffs[level] = [
            tf.slice(second_pass, [local_m // 2, 0, 0], [local_m // 2, local_n // 2, 1]),
            tf.slice(second_pass, [0, local_n // 2, 0], [local_m // 2, local_n // 2, 1]),
            tf.slice(second_pass, [local_m // 2, local_n // 2, 0],
                     [local_m // 2, local_n // 2, 1])
        ]

    for level in range(levels - 1, -1, -1):
        upper_half = tf.concat([last_level, coeffs[level][0]], 0)
        lower_half = tf.concat([coeffs[level][1], coeffs[level][2]], 0)

        last_level = tf.concat([upper_half, lower_half], 1)

    return last_level


def idwt1d(input_node, wavelet, levels=1):
    """
    Constructs a TF graph that computes the 1D inverse DWT for a given wavelet.

    Args:
        input_node (tf.placeholder):             Input signal. A 3D tensor with dimensions
                                                 as [batch, signal, channels]
        wavelet (tfwavelets.dwtcoeffs.Wavelet):  Wavelet object.
        levels (int):                            Number of levels.

    Returns:
        Output node of IDWT graph.
    """
    m, n = int(input_node.shape[0]), int(input_node.shape[1])

    first_n = n // (2 ** levels)
    last_level = tf.slice(input_node, [0, 0, 0], [m, first_n, 1])

    for level in range(levels - 1, -1 , -1):
        local_n = n // (2 ** level)

        detail = tf.slice(input_node, [0, local_n//2, 0], [m, local_n//2, 1])

        lowres_padded = upsample(last_level, odd=False)
        detail_padded = upsample(detail, odd=True)

        lowres_filtered = cyclic_conv1d_alt(lowres_padded, wavelet.recon_lp)
        detail_filtered = cyclic_conv1d_alt(detail_padded, wavelet.recon_hp)

        last_level = lowres_filtered + detail_filtered

    return last_level


def idwt2d(input_node, wavelet, levels=1):
    """
    Constructs a TF graph that computes the 2D inverse DWT for a given wavelet.

    Args:
        input_node (tf.placeholder):             Input signal. A 3D tensor with dimensions
                                                 as [rows, cols, channels]
        wavelet (tfwavelets.dwtcoeffs.Wavelet):  Wavelet object.
        levels (int):                            Number of levels.

    Returns:
        Output node of IDWT graph.
    """
    m, n = int(input_node.shape[0]), int(input_node.shape[1])
    first_m, first_n = m // (2 ** levels), n // (2 ** levels)

    last_level = tf.slice(input_node, [0, 0, 0], [first_m, first_n, 1])

    for level in range(levels - 1, -1, -1):
        local_m, local_n = m // (2 ** level), n // (2 ** level)

        # Extract detail spaces
        detail_tr = tf.slice(input_node, [local_m // 2, 0, 0],
                             [local_n // 2, local_m // 2, 1])
        detail_bl = tf.slice(input_node, [0, local_n // 2, 0],
                             [local_n // 2, local_m // 2, 1])
        detail_br = tf.slice(input_node, [local_n // 2, local_m // 2, 0],
                             [local_n // 2, local_m // 2, 1])

        # Construct image of this DWT level
        upper_half = tf.concat([last_level, detail_tr], 0)
        lower_half = tf.concat([detail_bl, detail_br], 0)

        this_level = tf.concat([upper_half, lower_half], 1)

        # First pass, corresponding to second pass in dwt2d
        first_pass = tf.transpose(
            idwt1d(
                tf.transpose(this_level, perm=[1, 0, 2]),
                wavelet,
                1
            ),
            perm=[1, 0, 2]
        )
        # Second pass, corresponding to first pass in dwt2d
        second_pass = idwt1d(first_pass, wavelet, 1)

        last_level = second_pass

    return last_level
