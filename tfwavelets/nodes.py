import tensorflow as tf
import numpy as np


def _adapt_filter(filter):
    # Add empty dimensions for batch size and channel num
    return np.expand_dims(np.expand_dims(filter, -1), -1)


def dwt1d(input_node, filter_coeffs, levels=1):
    # TODO: Check that level is a reasonable number
    # TODO: Check types

    coeffs = [None] * (levels + 1)

    last_level = input_node

    lp_adapted = _adapt_filter(filter_coeffs[0])
    hp_adapted = _adapt_filter(filter_coeffs[1])

    tf_lp = tf.constant(lp_adapted, dtype=tf.float32, shape=[len(filter_coeffs[0]), 1, 1])
    tf_hp = tf.constant(hp_adapted, dtype=tf.float32, shape=[len(filter_coeffs[1]), 1, 1])

    for level in range(levels):
        # TODO: Convert stride kwarg to tuple
        # TODO: Actual convolution, not correlation
        # TODO: Periodic extention, not zero-padding
        lp_res = tf.nn.conv1d(last_level, tf_lp, stride=2, padding="SAME")
        hp_res = tf.nn.conv1d(last_level, tf_hp, stride=2, padding="SAME")

        last_level = lp_res
        coeffs[levels - level] = hp_res

    coeffs[0] = last_level
    return tf.concat(coeffs, axis=1)

