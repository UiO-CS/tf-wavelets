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


def dwt2d(input_node, filter_coeffs, levels=1):
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
