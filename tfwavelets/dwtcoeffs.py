import numpy as np

# Haar wavelet
haar = (np.array([0.70710677, 0.70710677], dtype=np.float32),
        np.array([0.70710677, -0.70710677], dtype=np.float32))

# Daubechies wavelets
db1 = haar
db2 = (np.array([-0.12940952255092145,
                  0.22414386804185735,
                  0.836516303737469,
                  0.48296291314469025], dtype=np.float32),
       np.array([-0.48296291314469025,
                  0.836516303737469,
                 -0.22414386804185735,
                 -0.12940952255092145], dtype=np.float32))
db3 = (np.array([ 0.035226291882100656,
                 -0.08544127388224149,
                 -0.13501102001039084,
                  0.4598775021193313,
                  0.8068915093133388,
                  0.3326705529509569], dtype=np.float32),
       np.array([-0.3326705529509569,
                  0.8068915093133388,
                 -0.4598775021193313,
                 -0.13501102001039084,
                  0.08544127388224149,
                  0.035226291882100656], dtype=np.float32))
db4 = (np.array([-0.010597401784997278,
                  0.032883011666982945,
                  0.030841381835986965,
                 -0.18703481171888114,
                 -0.02798376941698385,
                  0.6308807679295904,
                  0.7148465705525415,
                  0.23037781330885523], dtype=np.float32),
       np.array([-0.23037781330885523,
                  0.7148465705525415,
                 -0.6308807679295904,
                 -0.02798376941698385,
                  0.18703481171888114,
                  0.030841381835986965,
                 -0.032883011666982945,
                 -0.010597401784997278], dtype=np.float32))


def edge_matrices(coeffs, zero):
    """Computes the submatrices needed at the ends for circular convolution.

    Args:
        coeffs: 1d numpy array with the filter coefficients, ordered by
                filter index.
        zero:   Index in `coeffs` where the zero-indexed filter coefficient is.

    Returns:
        Tuple of 2d-arrays, (top-left, top-right, bottom-left, bottom-right).
    """
    if not isinstance(coeffs, np.ndarray):
        coeffs = np.array(coeffs)

    n, = coeffs.shape
    coeffs = coeffs[::-1]

    # Some padding is necesssary to keep the submatrices
    # from having having columns in common
    padding = max((zero, n-zero-1))
    matrix_size = n+padding
    filter_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)
    negative = coeffs[-(zero+1):] # negative indexed filter coeffs (and 0)
    positive = coeffs[:-(zero+1)] # filter coeffs with strictly positive indeces

    # Insert first row
    filter_matrix[0, :len(negative)] = negative
    filter_matrix[0, -len(positive):] = positive

    # Cycle previous row to compute the entire filter matrix
    for i in range(1, matrix_size):
        filter_matrix[i,:] = np.roll(filter_matrix[i-1,:], 1)


    # TODO: Indexing not thoroughly tested
    num_pos = len(positive)
    num_neg = len(negative)
    top_left = filter_matrix[:num_pos, :(num_pos+num_neg-1)]
    top_right = filter_matrix[:num_pos, -num_pos:]
    bottom_left = filter_matrix[-num_neg+1:, :num_neg-1]
    bottom_right = filter_matrix[-num_neg+1:, -(num_pos+num_neg-1):]

    return(top_left, top_right, bottom_left, bottom_right)
