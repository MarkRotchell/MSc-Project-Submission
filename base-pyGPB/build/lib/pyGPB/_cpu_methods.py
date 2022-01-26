import numpy as np
import numba

from scipy.fft._pocketfft.pypocketfft import r2c, c2r

from cmath import exp, pi

from pyGPB._tools import _good_size_for_real_FFT

DEFAULT_MAX_FOR_SERIAL_DP = 128

"""### Naive Approach"""


@numba.jit
def _GPB_Naive(probs, weights):
    """Compute the pmf vector of the Generalised Poisson Binomial Distribution
    in two parameter form. Uses a naive, or brute-force, approach of enumerating
    all possible outcomes and collecting the probabilities.

    Private Function without any type checking or input validation. Assumes
    probs and weights are the same length

    :param probs: (numpy.ndarray, 1D, dtype=numpy.float64): success probabilities
    :param weights: (numpy.ndarray, 1D, dtype=numpy.int32): positive success weights
    :return: (numpy.ndarray, 1D, dtype=numpy.float64) the pmf vector  
    """
    a = np.zeros(weights.sum() + 1)
    M = len(probs)
    for i in range(2 ** M):
        p = 1.0
        w = 0
        for j in range(M):
            flag = ((i >> j) & 1)
            w += flag * weights[j]
            p *= abs(1 - probs[j] - flag)
        a[w] += p
    return a


"""### Characteristic Function Approach"""


@numba.jit
def _char_func(probs, weights):
    N = weights.sum() + 1
    M = len(weights)
    res = np.zeros(N // 2 + 1, dtype=np.complex128)
    for k in range(N // 2 + 1):
        x = 1
        for j in range(M):
            x *= (1 + probs[j] * (exp(-2j * pi * k * weights[j] / N) - 1))
        res[k] = x
    return res


def _GPB_CF(probs, weights):
    """Compute the pmf vector of the Generalised Poisson Binomial Distribution
    in two parameter form. First calculates the characteristic function and 
    then uses an inverse fourier transform to turn that back into a pmf.

    Private Function without any type checking or input validation. Assumes
    probs and weights are the same length

    :param probs: (numpy.ndarray, 1D, dtype=numpy.float64): success probabilities
    :param weights: (numpy.ndarray, 1D, dtype=numpy.int32): positive success weights
    :return: (numpy.ndarray, 1D, dtype=numpy.float64) the pmf vector  cd
    """
    return c2r(_char_func(probs, weights), (-1,), weights.sum() + 1, False, 2, None, 0)


"""### Serial Dynamic Programming Approach"""


@numba.njit
def _GPB_DP(probs, weights):
    """Compute the pmf vector of the Generalised Poisson Binomial Distribution
    in two parameter form. Uses a Dynamic programming approach

    Private Function without any type checking or input validation. Assumes
    probs and weights are the same length

    :param probs: (numpy.ndarray, 1D, dtype=numpy.float64): success probabilities
    :param weights: (numpy.ndarray, 1D, dtype=numpy.int32): positive success weights
    :return: (numpy.ndarray, 1D, dtype=numpy.float64) the pmf vector  
    """
    probs = probs[np.argsort(weights)]
    weights = np.sort(weights)
    a = np.zeros(weights.sum() + 1, dtype=probs.dtype)
    a[0] = 1.0
    n = 1
    for p, w in zip(probs, weights):
        a[w:n + w] = a[w:n + w] * (1 - p) + a[:n] * p
        a[:w] = a[:w] * (1 - p)
        n += w
    return a


"""### Divide and Conquer FFT Approach"""


def _convolve_1d(a, b):
    """Compute the 1D convolution of two vectors and return the result

    Private Function without any type checking or input validation. Assumes
    probs and weights are the same length

    :param probs: (numpy.ndarray, 1D, dtype=numpy.float64): success probabilities
    :param weights: (numpy.ndarray, 1D, dtype=numpy.int32): positive success weights
    :return: (numpy.ndarray, 1D, dtype=numpy.float64) the pmf vector  
    """
    n_a = a.shape[0]
    n_b = b.shape[0]
    n = n_a + n_b - 1
    m = _good_size_for_real_FFT(n)
    a_pad = np.zeros(m)
    b_pad = np.zeros(m)
    a_pad[:n_a] = a
    b_pad[:n_b] = b
    with numba.objmode(y='float64[:]'):
        y = c2r(r2c(a_pad, (0,), True, 0, None, 1) * r2c(b_pad, (0,), True, 0, None, 1), (-1,), m, False, 2, None, 1)
    return y[:n]


def _GPB_DC_FFT(probs, weights):
    """Compute the pmf vector of the Generalised Poisson Binomial Distribution
    in two parameter form. Uses a recursive divide and conquer approach with 
    Fast Fourier Transforms

    Private Function without any type checking or input validation. Assumes
    probs and weights are the same length

    :param probs: (numpy.ndarray, 1D, dtype=numpy.float64): success probabilities
    :param weights: (numpy.ndarray, 1D, dtype=numpy.int32): positive success weights
    :return: (numpy.ndarray, 1D, dtype=numpy.float64) the pmf vector  
    """
    if len(probs) == 1:
        prob, weight = probs[0], weights[0]
        array = np.zeros(weight + 1)
        array[0] = 1 - prob
        array[weight] = prob
        return array
    else:
        mid = len(probs) // 2
        left = _GPB_DC_FFT(probs[:mid], weights[:mid])
        right = _GPB_DC_FFT(probs[mid:], weights[mid:])
        return _convolve_1d(left, right)


"""### Combination of DP and FFT approaches, using DP for small M"""


def _GPB_DP_DC_FFT_Combo(probs, weights, max_for_serial_dp=DEFAULT_MAX_FOR_SERIAL_DP):
    """Compute the pmf vector of the Generalised Poisson Binomial Distribution
    in two parameter form. Uses a combination of dynamic programming for small 
    length inputs, and FFT approach for longer length inputs

    Private Function without any type checking or input validation. Assumes
    probs and weights are the same length

    :param probs: (numpy.ndarray, 1D, dtype=numpy.float64): success probabilities
    :param weights: (numpy.ndarray, 1D, dtype=numpy.int32): positive success weights
    :return: (numpy.ndarray, 1D, dtype=numpy.float64) the pmf vector  
    """
    if probs.shape[0] < max_for_serial_dp:
        return _GPB_DP(probs, weights)
    else:
        mid = len(probs) // 2
        left = _GPB_DP_DC_FFT_Combo(probs[:mid], weights[:mid], max_for_serial_dp)
        right = _GPB_DP_DC_FFT_Combo(probs[mid:], weights[mid:], max_for_serial_dp)
        return _convolve_1d(left, right)


_available_methods = {
    'Naive': _GPB_Naive,
    'CF': _GPB_CF,
    'DP': _GPB_DP,
    'FFT': _GPB_DC_FFT,
    'Fastest_CPU': _GPB_DP_DC_FFT_Combo,
}
