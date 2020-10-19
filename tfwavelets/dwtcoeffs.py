"""
The 'dwtcoeffs' module contains predefined wavelets, as well as the classes necessary to
create more user-defined wavelets.

Wavelets are defined by the Wavelet class. A Wavelet object mainly consists of four Filter
objects (defined by the Filter class) representing the decomposition and reconstruction
low pass and high pass filters.

Examples:
    You can define your own wavelet by creating four filters, and combining them to a wavelet:

    >>> decomp_lp = Filter([1 / np.sqrt(2), 1 / np.sqrt(2)], 0)
    >>> decomp_hp = Filter([1 / np.sqrt(2), -1 / np.sqrt(2)], 1)
    >>> recon_lp = Filter([1 / np.sqrt(2), 1 / np.sqrt(2)], 0)
    >>> recon_hp = Filter([-1 / np.sqrt(2), 1 / np.sqrt(2)], 1)
    >>> haar = Wavelet(decomp_lp, decomp_hp, recon_lp, recon_hp)

"""

import numpy as np
import tensorflow as tf
from tfwavelets.utils import adapt_filter, to_tf_mat


class Filter:
    """
    Class representing a filter.

    Attributes:
        coeffs (tf.constant):      Filter coefficients
        zero (int):                Origin of filter (which index of coeffs array is
                                   actually indexed as 0).
        edge_matrices (iterable):  List of edge matrices, used for circular convolution.
                                   Stored as 3D TF tensors (constants).
    """


    def __init__(self, coeffs, zero, dtype=tf.float32):
        """
        Create a filter based on given filter coefficients

        Args:
            coeffs (np.ndarray):       Filter coefficients
            zero (int):                Origin of filter (which index of coeffs array is
                                       actually indexed as 0).
        """
        self.dtype=dtype;
        if dtype == tf.float32:
            self.npdtype = np.float32;
        elif dtype == tf.float64:
            self.npdtype = np.float64;

        self.coeffs = tf.constant(adapt_filter(coeffs), dtype=self.dtype)

        if not isinstance(coeffs, np.ndarray):
            coeffs = np.array(self.coeffs)
        self._coeffs = coeffs.astype(self.npdtype)

        self.zero = zero

        self.edge_matrices = to_tf_mat(self._edge_matrices(), dtype=self.dtype)


    def __getitem__(self, item):
        """
        Returns filter coefficients at requested indeces. Indeces are offset by the filter
        origin

        Args:
            item (int or slice):    Item(s) to get

        Returns:
            np.ndarray: Item(s) at specified place(s)
        """
        if isinstance(item, slice):
            return self._coeffs.__getitem__(
                slice(item.start + self.zero, item.stop + self.zero, item.step)
            )
        else:
            return self._coeffs.__getitem__(item + self.zero)


    def num_pos(self):
        """
        Number of positive indexed coefficients in filter, including the origin. Ie,
        strictly speaking it's the number of non-negative indexed coefficients.

        Returns:
            int: Number of positive indexed coefficients in filter.
        """
        return len(self._coeffs) - self.zero


    def num_neg(self):
        """
        Number of negative indexed coefficients, excluding the origin.

        Returns:
            int: Number of negative indexed coefficients
        """
        return self.zero


    def _edge_matrices(self):
        """Computes the submatrices needed at the ends for circular convolution.

        Returns:
            Tuple of 2d-arrays, (top-left, top-right, bottom-left, bottom-right).
        """
        if not isinstance(self._coeffs, np.ndarray):
            self._coeffs = np.array(self._coeffs)

        n, = self._coeffs.shape
        self._coeffs = self._coeffs[::-1]

        # Some padding is necesssary to keep the submatrices
        # from having having columns in common
        padding = max((self.zero, n - self.zero - 1))
        matrix_size = n + padding
        filter_matrix = np.zeros((matrix_size, matrix_size), dtype=self.npdtype)
        negative = self._coeffs[
                   -(self.zero + 1):]  # negative indexed filter coeffs (and 0)
        positive = self._coeffs[
                   :-(self.zero + 1)]  # filter coeffs with strictly positive indeces

        # Insert first row
        filter_matrix[0, :len(negative)] = negative

        # Because -0 == 0, a length of 0 makes it impossible to broadcast
        # (nor is is necessary)
        if len(positive) > 0:
            filter_matrix[0, -len(positive):] = positive

        # Cycle previous row to compute the entire filter matrix
        for i in range(1, matrix_size):
            filter_matrix[i, :] = np.roll(filter_matrix[i - 1, :], 1)

        # TODO: Indexing not thoroughly tested
        num_pos = len(positive)
        num_neg = len(negative)
        top_left = filter_matrix[:num_pos, :(num_pos + num_neg - 1)]
        top_right = filter_matrix[:num_pos, -num_pos:]
        bottom_left = filter_matrix[-num_neg + 1:, :num_neg - 1]
        bottom_right = filter_matrix[-num_neg + 1:, -(num_pos + num_neg - 1):]

        # Indexing wrong when there are no negative indexed coefficients
        if num_neg == 1:
            bottom_left = np.zeros((0, 0), dtype=self.npdtype)
            bottom_right = np.zeros((0, 0), dtype=self.npdtype)

        return top_left, top_right, bottom_left, bottom_right


class TrainableFilter(Filter):
    """
    Class representing a trainable filter.

    Attributes:
        coeffs (tf.Variable):      Filter coefficients
        zero (int):                Origin of filter (which index of coeffs array is
                                   actually indexed as 0).
    """


    def __init__(self, initial_coeffs, zero, dtype=tf.float32, name=None):
        """
        Create a trainable filter initialized with given filter coefficients

        Args:
            initial_coeffs (np.ndarray):    Initial filter coefficients
            zero (int):                     Origin of filter (which index of coeffs array
                                            is actually indexed as 0).
            name (str):                     Optional. Name of tf variable created to hold
                                            the filter coeffs.
        """
        super().__init__(initial_coeffs, zero, dtype=dtype)

        self.coeffs = tf.Variable(
            initial_value=adapt_filter(initial_coeffs),
            trainable=True,
            name=name,
            dtype=dtype,
            constraint=tf.keras.constraints.max_norm(np.sqrt(2), [1, 2])
        )

        # Erase stuff that will be invalid once the filter coeffs has changed
        self._coeffs = [None]*len(self._coeffs)
        self.edge_matrices = None


class Wavelet:
    """
    Class representing a wavelet.

    Attributes:
        decomp_lp (Filter):    Filter coefficients for decomposition low pass filter
        decomp_hp (Filter):    Filter coefficients for decomposition high pass filter
        recon_lp (Filter):     Filter coefficients for reconstruction low pass filter
        recon_hp (Filter):     Filter coefficients for reconstruction high pass filter
    """


    def __init__(self, decomp_lp, decomp_hp, recon_lp, recon_hp):
        """
        Create a new wavelet based on specified filters

        Args:
            decomp_lp (Filter):    Filter coefficients for decomposition low pass filter
            decomp_hp (Filter):    Filter coefficients for decomposition high pass filter
            recon_lp (Filter):     Filter coefficients for reconstruction low pass filter
            recon_hp (Filter):     Filter coefficients for reconstruction high pass filter
        """
        self.decomp_lp = decomp_lp
        self.decomp_hp = decomp_hp
        self.recon_lp = recon_lp
        self.recon_hp = recon_hp


class TrainableWavelet(Wavelet):
    """
    Class representing a trainable wavelet

    Attributes:
        decomp_lp (TrainableFilter):    Filter coefficients for decomposition low pass filter
        decomp_hp (TrainableFilter):    Filter coefficients for decomposition high pass filter
        recon_lp (TrainableFilter):     Filter coefficients for reconstruction low pass filter
        recon_hp (TrainableFilter):     Filter coefficients for reconstruction high pass filter
    """


    def __init__(self, wavelet):
        """
        Create a new trainable wavelet initialized as specified wavelet

        Args:
            wavelet (Wavelet):          Starting point for the trainable wavelet
        """
        super().__init__(
            TrainableFilter(wavelet.decomp_lp._coeffs, wavelet.decomp_lp.zero),
            TrainableFilter(wavelet.decomp_hp._coeffs, wavelet.decomp_hp.zero),
            TrainableFilter(wavelet.recon_lp._coeffs, wavelet.recon_lp.zero),
            TrainableFilter(wavelet.recon_hp._coeffs, wavelet.recon_hp.zero)
        )


# Haar wavelet
haar = Wavelet(
    Filter(np.array([0.70710677, 0.70710677]), 1),
    Filter(np.array([-0.70710677, 0.70710677]), 0),
    Filter(np.array([0.70710677, 0.70710677]), 0),
    Filter(np.array([0.70710677, -0.70710677]), 1),
)

# Daubechies wavelets
db1 = haar
db2 = Wavelet(
    Filter(np.array([-0.12940952255092145,
                     0.22414386804185735,
                     0.836516303737469,
                     0.48296291314469025]), 3),
    Filter(np.array([-0.48296291314469025,
                     0.836516303737469,
                     -0.22414386804185735,
                     -0.12940952255092145]), 0),
    Filter(np.array([0.48296291314469025,
                     0.836516303737469,
                     0.22414386804185735,
                     -0.12940952255092145]), 0),
    Filter(np.array([-0.12940952255092145,
                     -0.22414386804185735,
                     0.836516303737469,
                     -0.48296291314469025]), 3)
)
db3 = Wavelet(
    Filter(np.array([0.035226291882100656,
                    -0.08544127388224149,
                    -0.13501102001039084,
                    0.4598775021193313,
                    0.8068915093133388,
                    0.3326705529509569]), 5),
    Filter(np.array([-0.3326705529509569,
                    0.8068915093133388,
                    -0.4598775021193313,
                    -0.13501102001039084,
                    0.08544127388224149,
                    0.035226291882100656]), 0),
    Filter(np.array([0.3326705529509569,
                    0.8068915093133388,
                    0.4598775021193313,
                    -0.13501102001039084,
                    -0.08544127388224149,
                    0.035226291882100656]), 0),
    Filter(np.array([0.035226291882100656,
                    0.08544127388224149,
                    -0.13501102001039084,
                    -0.4598775021193313,
                    0.8068915093133388,
                    -0.3326705529509569]), 5)
)
db4 = Wavelet(
    Filter(np.array([-0.010597401784997278,
                    0.032883011666982945,
                    0.030841381835986965,
                    -0.18703481171888114,
                    -0.02798376941698385,
                    0.6308807679295904,
                    0.7148465705525415,
                    0.23037781330885523]), 7),
    Filter(np.array([-0.23037781330885523,
                    0.7148465705525415,
                    -0.6308807679295904,
                    -0.02798376941698385,
                    0.18703481171888114,
                    0.030841381835986965,
                    -0.032883011666982945,
                    -0.010597401784997278]), 0),
    Filter(np.array([0.23037781330885523,
                    0.7148465705525415,
                    0.6308807679295904,
                    -0.02798376941698385,
                    -0.18703481171888114,
                    0.030841381835986965,
                    0.032883011666982945,
                    -0.010597401784997278]), 0),
    Filter(np.array([-0.010597401784997278,
                    -0.032883011666982945,
                    0.030841381835986965,
                    0.18703481171888114,
                    -0.02798376941698385,
                    -0.6308807679295904,
                    0.7148465705525415,
                    -0.23037781330885523]), 7)
)

def get_wavelet(wavelet_name, dtype=tf.float32):
    """
    Get a wavelet based on the wavelets name.

    Args:
        wavelet_name (str): Name of the wavelet ('haar', 'db1', 'db2', 'db3' or 'db4').

    Returns:
        A wavelet object. If the wavelet name is not recognized, it returns None.
    """
    wname = wavelet_name.lower()
    if wname == 'db1' or wname == 'haar':
        # Haar wavelet
        a = 1/np.sqrt(2);
        haar = Wavelet(
                       Filter(np.array([a,  a]), 1, dtype=dtype),
                       Filter(np.array([-a, a]), 0, dtype=dtype),
                       Filter(np.array([a,  a]), 0, dtype=dtype),
                       Filter(np.array([a, -a]), 1, dtype=dtype),
                        )
        return haar
    elif wname == 'db2':
        db2 = Wavelet(
            Filter(np.array([-0.12940952255092145,
                             0.22414386804185735,
                             0.836516303737469,
                             0.48296291314469025]), 3, dtype=dtype),
            Filter(np.array([-0.48296291314469025,
                             0.836516303737469,
                             -0.22414386804185735,
                             -0.12940952255092145]), 0, dtype=dtype),
            Filter(np.array([0.48296291314469025,
                             0.836516303737469,
                             0.22414386804185735,
                             -0.12940952255092145]), 0, dtype=dtype),
            Filter(np.array([-0.12940952255092145,
                             -0.22414386804185735,
                             0.836516303737469,
                             -0.48296291314469025]), 3, dtype=dtype)
        )
        return db2
    elif wname == 'db3':
        db3 = Wavelet(
            Filter(np.array([0.035226291882100656,
                            -0.08544127388224149,
                            -0.13501102001039084,
                            0.4598775021193313,
                            0.8068915093133388,
                            0.3326705529509569]), 5, dtype=dtype),
            Filter(np.array([-0.3326705529509569,
                            0.8068915093133388,
                            -0.4598775021193313,
                            -0.13501102001039084,
                            0.08544127388224149,
                            0.035226291882100656]), 0, dtype=dtype),
            Filter(np.array([0.3326705529509569,
                            0.8068915093133388,
                            0.4598775021193313,
                            -0.13501102001039084,
                            -0.08544127388224149,
                            0.035226291882100656]), 0, dtype=dtype),
            Filter(np.array([0.035226291882100656,
                            0.08544127388224149,
                            -0.13501102001039084,
                            -0.4598775021193313,
                            0.8068915093133388,
                            -0.3326705529509569]), 5, dtype=dtype)
        )
        return db3
    elif wname == 'db4':
        db4 = Wavelet(
            Filter(np.array([-0.010597401784997278,
                            0.032883011666982945,
                            0.030841381835986965,
                            -0.18703481171888114,
                            -0.02798376941698385,
                            0.6308807679295904,
                            0.7148465705525415,
                            0.23037781330885523]), 7, dtype=dtype),
            Filter(np.array([-0.23037781330885523,
                            0.7148465705525415,
                            -0.6308807679295904,
                            -0.02798376941698385,
                            0.18703481171888114,
                            0.030841381835986965,
                            -0.032883011666982945,
                            -0.010597401784997278]), 0, dtype=dtype),
            Filter(np.array([0.23037781330885523,
                            0.7148465705525415,
                            0.6308807679295904,
                            -0.02798376941698385,
                            -0.18703481171888114,
                            0.030841381835986965,
                            0.032883011666982945,
                            -0.010597401784997278]), 0, dtype=dtype),
            Filter(np.array([-0.010597401784997278,
                            -0.032883011666982945,
                            0.030841381835986965,
                            0.18703481171888114,
                            -0.02798376941698385,
                            -0.6308807679295904,
                            0.7148465705525415,
                            -0.23037781330885523]), 7, dtype=dtype)
        )
        return db4
    else:
        return None

