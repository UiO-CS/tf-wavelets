import tfwavelets as tfw
import numpy as np


def check_orthonormality_1d(wavelet, tol=1e-5, N=8):
    matrix = np.zeros((N, N))

    for i in range(N):
        unit = np.zeros(N)
        unit[i] = 1

        matrix[:, i] = tfw.wrappers.dwt1d(unit, wavelet)

    error1 = np.sum(np.abs(matrix.T @ matrix - np.eye(N))) / N
    error2 = np.sum(np.abs(matrix @ matrix.T - np.eye(N))) / N
    assert error1 < tol, "Mean error: %g" % error1
    assert error2 < tol, "Mean error: %g" % error2


def check_linearity_1d(wavelet, tol=1e-5, N=256):
    x1 = np.random.random(N)
    x2 = np.random.random(N)

    c1 = np.random.random(1)
    c2 = np.random.random(1)

    test1 = tfw.wrappers.dwt1d(c1 * x1 + c2 * x2)
    test2 = c1 * tfw.wrappers.dwt1d(x1) + c2 * tfw.wrappers.dwt1d(x2)

    error = np.sum(np.abs(test1 - test2)) / (N ** 2)
    assert error < tol, "Mean error: %g" % error


def check_linearity_2d(wavelet, tol=1e-5, N=256):
    x1 = np.random.random((N, N))
    x2 = np.random.random((N, N))

    c1 = np.random.random(1)
    c2 = np.random.random(1)

    test1 = tfw.wrappers.dwt2d(c1 * x1 + c2 * x2)
    test2 = c1 * tfw.wrappers.dwt2d(x1) + c2 * tfw.wrappers.dwt2d(x2)

    error = np.sum(np.abs(test1 - test2)) / (N ** 2)
    assert error < tol, "Mean error: %g" % error


def test_ortho_haar():
    check_orthonormality_1d("haar")


def test_ortho_db2():
    check_orthonormality_1d("db2")


def test_linear_haar_1d():
    check_linearity_1d("haar")


def test_linear_db2_1d():
    check_linearity_1d("db2")


def test_linear_haar_2d():
    check_linearity_2d("haar")


def test_linear_db2_2d():
    check_linearity_2d("db2")


if __name__ == "__main__":
    test_linear_db2_1d()
    test_linear_haar_1d()
    test_linear_db2_2d()
    test_linear_haar_2d()
    test_ortho_db2()
    test_ortho_haar()
