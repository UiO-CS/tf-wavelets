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
