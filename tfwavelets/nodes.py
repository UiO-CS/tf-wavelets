import tensorflow as tf
import numpy as np

def dwt1d(input_node, filter_coeffs, levels=1):
    # TODO: Check that level is a reasonable number
    # TODO: Check types

    coeffs = [None]*(levels+1)

    last_level = input_node
    
    lp_adapted = np.expand_dims(filter_coeffs[0], -1)
    lp_adapted = np.expand_dims(lp_adapted, -1)

    hp_adapted = np.expand_dims(filter_coeffs[1], -1)
    hp_adapted = np.expand_dims(hp_adapted, -1)
    
    tf_lp = tf.constant(lp_adapted, dtype=tf.float32, shape=[len(filter_coeffs[0]),1,1])
    tf_hp = tf.constant(hp_adapted, dtype=tf.float32, shape=[len(filter_coeffs[1]),1,1])

    for level in range(levels):
        # TODO: Convert stride kwarg to tuple
        # TODO: Actual convolution, not correlation
        # TODO: Periodic extention, not zero-padding
        lp_res = tf.nn.conv1d(last_level, tf_lp, stride=2, padding="SAME")
        hp_res = tf.nn.conv1d(last_level, tf_hp, stride=2, padding="SAME")

        last_level = lp_res
        coeffs[levels-level] = hp_res


    coeffs[0] = last_level
    return tf.concat(coeffs, axis=1)

def cyclic_conv1d(input_node, kernel_node, tl_node, tr_node, bl_node, br_node):
    """Cyclic convolution

    Args:
        input_node:  Input signal (3-tensor [batch, width, in_channels])
        filter_node: Filter (3-tensor[filter_width, in_channels, out_channels]) 
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
                        [0, input_shape[1]-tr_shape[2], 0],
                        [-1, tr_shape[1], -1])
    bl_slice = tf.slice(input_node,
                        [0, 0, 0],
                        [-1, bl_shape[2], -1])
    br_slice = tf.slice(input_node,
                        [0, input_shape[1]-br_shape[2], 0],
                        [-1, br_shape[2], -1])

    # TODO: It just werks (It's the magic of the algorithm). i.e. Why do we have to transpose?
    tl = tl_node @ tf.transpose(tl_slice, perm=[2,1,0])
    tr = tr_node @ tf.transpose(tr_slice, perm=[2,1,0])
    bl = bl_node @ tf.transpose(bl_slice, perm=[2,1,0])
    br = br_node @ tf.transpose(br_slice, perm=[2,1,0])

    head = tf.transpose(tl + tr, perm=[2,1,0])
    tail = tf.transpose(bl + br, perm=[2,1,0])

    return tf.concat((head, inner, tail), axis=1)
