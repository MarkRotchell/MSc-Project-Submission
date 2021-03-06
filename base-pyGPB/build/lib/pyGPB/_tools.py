# -*- coding: utf-8 -*-
"""Untitled50.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1em-aitKF2ok24YIDJERHYTzxXhmHUZ2B
"""

import numba


@numba.njit(numba.int32(numba.int32), cache=True, fastmath=True)
def _good_size_for_real_FFT(n):
    """The smallest 5-smooth number that is greater than or equal to n
    A 5-smooth number is a number whose prime factors are all less than or equal to 5
    Fast Fourier Transforms are particularly fast when their length has small prime factors
    so it is generally more efficient to pad the length to the next n-smooth number
    rather than use the original length.
    Algorithm taken from
    https://github.com/scipy/scipy/blob/701ffcc8a6f04509d115aac5e5681c538b5265a2/scipy/fft/_pocketfft/pocketfft_hdronly.h
    :param n: (int) the number to round up to a 5-smooth number
    :return: (int) the 5-smooth number
    """

    if n <= 6:
        return n
    bestfac = 2 * n
    f5 = 1
    while f5 < bestfac:
        x = f5
        while x < n:
            x *= 2
        while True:
            if x < n:
                x *= 3
            elif x > n:
                if x < bestfac:
                    bestfac = x
                if x & 1:
                    break
                x >>= 1
            else:
                return n
        f5 *= 5
    return bestfac
