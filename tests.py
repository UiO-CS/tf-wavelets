import tfwavelets as tfw
import numpy as np


def check_orthonormality_1d(wavelet, tol=1e-5, N=8):
    matrix = np.zeros((N, N))

    for i in range(N):
        unit = np.zeros(N)
        unit[i] = 1

        matrix[:, i] = tfw.wrappers.dwt1d(unit, wavelet)

    error1 = np.mean(np.abs(matrix.T @ matrix - np.eye(N)))
    error2 = np.mean(np.abs(matrix @ matrix.T - np.eye(N)))
    assert error1 < tol, "Mean error: %g" % error1
    assert error2 < tol, "Mean error: %g" % error2


def check_linearity_1d(wavelet, tol=1e-5, N=256):
    x1 = np.random.random(N)
    x2 = np.random.random(N)

    c1 = np.random.random(1)
    c2 = np.random.random(1)

    test1 = tfw.wrappers.dwt1d(c1 * x1 + c2 * x2)
    test2 = c1 * tfw.wrappers.dwt1d(x1) + c2 * tfw.wrappers.dwt1d(x2)

    error = np.mean(np.abs(test1 - test2))
    assert error < tol, "Mean error: %g" % error


def check_linearity_2d(wavelet, tol=1e-5, N=256):
    x1 = np.random.random((N, N))
    x2 = np.random.random((N, N))

    c1 = np.random.random(1)
    c2 = np.random.random(1)

    test1 = tfw.wrappers.dwt2d(c1 * x1 + c2 * x2)
    test2 = c1 * tfw.wrappers.dwt2d(x1) + c2 * tfw.wrappers.dwt2d(x2)

    error = np.mean(np.abs(test1 - test2))
    assert error < tol, "Mean error: %g" % error


def check_inverse_1d(wavelet, levels=1, tol=1e-4, N=256):
    signal = np.random.random(N)

    reconstructed = tfw.wrappers.idwt1d(
        tfw.wrappers.dwt1d(signal, levels=levels),
        levels=levels
    )

    error = np.mean(np.abs(signal - reconstructed))
    assert error < tol, "Mean error: %g" % error


def check_inverse_2d(wavelet, levels=1, tol=1e-4, N=256):
    signal = np.random.random((N, N))

    reconstructed = tfw.wrappers.idwt2d(
        tfw.wrappers.dwt2d(signal, levels=levels),
        levels=levels
    )

    error = np.mean(np.abs(signal - reconstructed))
    assert error < tol, "Mean error: %g" % error


def test_ortho_haar():
    check_orthonormality_1d("haar")

def test_linear_haar_1d():
    check_linearity_1d("haar")

def test_linear_haar_2d():
    check_linearity_2d("haar")

def test_inverse_haar_1d():
    check_inverse_1d("haar", levels=1)

def test_inverse_haar_1d_level2():
    check_inverse_1d("haar", levels=2)

def test_inverse_haar_2d():
    check_inverse_2d("haar", levels=2)

def test_ortho_db2():
    check_orthonormality_1d("db2")

def test_linear_db2_2d():
    check_linearity_2d("db2")

def test_linear_db2_1d():
    check_linearity_1d("db2")

def test_inverse_db2_1d():
    check_inverse_1d("db2", levels=1)

def test_inverse_db2_1d_level2():
    check_inverse_1d("db2", levels=2)

def test_inverse_db2_2d():
    check_inverse_2d("db2", levels=2)


def test_ortho_db3():
    check_orthonormality_1d("db3")

def test_linear_db3_2d():
    check_linearity_2d("db3")

def test_linear_db3_1d():
    check_linearity_1d("db3")

def test_inverse_db3_1d():
    check_inverse_1d("db3", levels=1)

def test_inverse_db3_1d_level2():
    check_inverse_1d("db3", levels=2)

def test_inverse_db3_2d():
    check_inverse_2d("db3", levels=2)


def test_ortho_db4():
    check_orthonormality_1d("db4")

def test_linear_db4_2d():
    check_linearity_2d("db4")

def test_linear_db4_1d():
    check_linearity_1d("db4")

def test_inverse_db4_1d():
    check_inverse_1d("db4", levels=1)

def test_inverse_db4_1d_level2():
    check_inverse_1d("db4", levels=2)

def test_inverse_db4_2d():
    check_inverse_2d("db4", levels=2)
