import numpy as np
import cupy as cp
from pyGPB._tools import _good_size_for_real_FFT

from cupy.cuda import cufft
from math import ceil, log2, sqrt

attributes = cp.cuda.Device().attributes
MAX_THREADS_PER_BLOCK = int(attributes['MaxThreadsPerBlock'])
MAX_THREADS_PER_MULTI_PROCESSOR = int(attributes['MaxThreadsPerMultiProcessor'])
MULTIPROCESSOR_COUNT = int(attributes['MultiProcessorCount'])
MAX_BLOCKS_PER_GRID_GROUP = int(MULTIPROCESSOR_COUNT * MAX_THREADS_PER_MULTI_PROCESSOR / MAX_THREADS_PER_BLOCK)
THREADS_PER_WARP = int(attributes['WarpSize'])
MAX_WARPS_PER_BLOCK = int(MAX_THREADS_PER_BLOCK // THREADS_PER_WARP)

"""### DP Approach"""

DP_Kernel_CUDA_C_Code = r'''
#include <cooperative_groups.h>
using namespace cooperative_groups;
extern "C" __global__
void _DP_Kernel(const double* p, const int* w, const int m, double* x1, double* x2) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int n = 0;
    unsigned int offset;
    unsigned int k;
    double* t;
    int w_j;
    double p_j;
    grid_group grid = this_grid();
    for (unsigned int j=0; j<m; ++j){
        w_j = w[j];
        p_j = p[j];
        n += w_j;
        offset = 0;
        while (offset <= n) {
            k = offset+i;
            if (k < w_j) {
                x2[k] = x1[k] * (1-p_j);
            } else if (k <= n) {
                x2[k] = x1[k] * (1-p_j)  + p_j * x1[k - w_j];
            };
            offset += blockDim.x * gridDim.x;
        };
        grid.sync();      
        t = x2;
        x2 = x1;
        x1 = t;        
    };
}
'''

_DP_Kernel = cp.RawKernel(DP_Kernel_CUDA_C_Code, '_DP_Kernel', enable_cooperative_groups=True)


def _GPB_DP_CUDA(probs, weights):
    """Compute the pmf vector of the Generalised Poisson Binomial Distribution
    in two parameter form. Uses a Dynamic Programming approach via CUDA with
    cooperative groups

    Private Function without any type checking or input validation. Assumes
    probs and weights are the same length

    :param probs: (numpy.ndarray, 1D, dtype=numpy.float64): success probabilities
    :param weights: (numpy.ndarray, 1D, dtype=numpy.int32): positive success weights
    :return: (cupy.ndarray, 1D, dtype=numpy.float64) the pmf vector  
    """
    p = cp.array(probs[np.argsort(weights)], dtype=np.float64)
    w = cp.array(np.sort(weights), dtype=np.int32)
    N = int(weights.sum()) + 1
    M = weights.shape[0]

    threads_per_block = MAX_THREADS_PER_BLOCK
    blocks_per_grid = min((N + threads_per_block - 1) // threads_per_block, MAX_BLOCKS_PER_GRID_GROUP)

    x1 = cp.zeros(N, dtype=np.float64)
    x2 = cp.zeros(N, dtype=np.float64)
    x1[0] = 1.0

    _DP_Kernel((blocks_per_grid,), (threads_per_block,), (p, w, M, x1, x2))
    if M % 2:
        return x2
    else:
        return x1


"""### DC-FFT + DP Approach"""


def _rfft_rows_gpu(a, n):
    """Perform a real-to-complex 1D FFT on each row of an array
    :param a: (cupy.ndarray, dtype=float64): the array of real numbers
    :param n: (int): the length of the FFT, nb all rows will be transformed to the same length
    :return: (cupy.ndarray, dtype=complex128) the fourier-transformed array    
    """
    if a.shape[-1] < n:
        a = cp.pad(a, ((0, 0), (0, n - a.shape[-1])))

    batch = a.shape[0]

    plan = cufft.Plan1d(n, cufft.CUFFT_D2Z, batch, devices=None)

    out = cp.empty((batch, n // 2 + 1), np.complex128)

    plan._single_gpu_fft(a, out, direction=cufft.CUFFT_FORWARD)

    return out


def _irfft_rows_gpu(a, n):
    """Perform a complex-to-real inverse 1D FFT on each row of an array
    :param a: (cupy.ndarray, 2D, dtype=complex128): the array of complex numbers
    :param n: (int): the length of the FFT, nb all rows will be transformed to the same length
    :return: (cupy.ndarray, 2D, dtype=float64) the inverse-fourier-transformed array    
    """
    batch = a.shape[0]

    plan = cufft.Plan1d(n, cufft.CUFFT_Z2D, batch, devices=None)

    out = cp.empty((batch, n), np.float64)

    plan._single_gpu_fft(a, out, direction=cufft.CUFFT_INVERSE)

    out /= n

    return out


def _omnivolve_rows(a, N):
    """Perform omnivolution (convolution of a set) on the rows of an array
    :param a: (cupy.ndarray, 2D, dtype=float64): the array of rows of real numbers 
    :param N: (int): the length of the final vector
    :return: (cupy.ndarray, 1D, dtype=float64) the omnivoled vector 
    """
    n = a.shape[1] * 2 - 1

    while a.shape[0] > 1:
        m = _good_size_for_real_FFT(n)

        b = _rfft_rows_gpu(a, m)

        mid = b.shape[0] // 2
        b[mid:][:mid] *= b[:mid]
        b = b[mid:].copy()

        a = _irfft_rows_gpu(b, m)[:, :n]

        n = 2 * n - 1

    return a[0, :N]


DP_2D_Kernel_CUDA_C_Code = r'''
extern "C" __global__
void _DP_Kernel_2D(const double* x, const double* p, const int* w, const int* n, double* z) {
    int row = blockIdx.x;
    int row_len = blockDim.x * gridDim.y;
    int col = blockDim.x * blockIdx.y + threadIdx.x;
    int weight = w[row];
    double prob = p[row];
    int n_pmf = n[row];
    int idx = row * row_len + col;
    if (weight == 0) {
          z[idx] = x[idx];
    } else if (col < weight) {
          z[idx] = x[idx] * (1-prob);
    } else if (col <= n_pmf) {
          z[idx] = x[idx] * (1-prob)  + prob * x[idx - weight];
    }
    
}
'''

_DP_Kernel_2D = cp.RawKernel(DP_2D_Kernel_CUDA_C_Code, '_DP_Kernel_2D')


def _GPB_DP_DC_FFT_CUDA(probs, weights):
    """Compute the pmf vector of the Generalised Poisson Binomial Distribution
    in two parameter form. Uses a mixture of 2D Dynamic Programming and Divide-and
    -conquer approach with FFTs.

    Private Function without any type checking or input validation. Assumes
    probs and weights are the same length

    :param probs: (numpy.ndarray, 1D, dtype=numpy.float64): success probabilities
    :param weights: (numpy.ndarray, 1D, dtype=numpy.int32): positive success weights
    :return: (cupy.ndarray, 1D, dtype=numpy.float64) the pmf vector  
    """
    n_items = weights.shape[0]

    n_DP_rows = 2 ** max(ceil(log2(n_items)) - 4, 1)

    # number of items once padded to a multiple of chunk_size
    n_items_per_dp_pmf = ((n_items + n_DP_rows - 1) // n_DP_rows)
    n_items_padded = n_items_per_dp_pmf * n_DP_rows

    # sort and pad to correct length
    pad_size = (n_items_padded - n_items, 0)
    p = np.pad(probs[np.argsort(weights)], pad_size)
    w = np.pad(np.sort(weights), pad_size)

    # reshape to 2D
    w = w.reshape((-1, n_DP_rows))
    p = p.reshape((-1, n_DP_rows))

    # reverse contents of alternate rows as heuristic for making columns equal total
    w[::2] = w[::2, ::-1]
    p[::2] = p[::2, ::-1]

    # running totals
    n = w.cumsum(axis=0)

    # most efficient when num threads and num block roughly equal
    # and num threads divides the size of a warp, subject to the 
    # maximum number of warps in a block

    oa_pmf_length = n[-1].max() + 1
    oa_pmf_area = oa_pmf_length * n_DP_rows

    warps_per_block = int(round(sqrt(oa_pmf_area) / THREADS_PER_WARP, 0))
    warps_per_block = max(min(warps_per_block, 32), 1)
    threads_per_block = warps_per_block * THREADS_PER_WARP
    blocks_per_oa_pmf = (oa_pmf_length + threads_per_block - 1) // threads_per_block

    # working arrays
    dp_pmf_length_padded = blocks_per_oa_pmf * threads_per_block
    working_array_1 = cp.zeros(shape=(n_DP_rows, dp_pmf_length_padded), dtype=np.float64)
    working_array_2 = cp.zeros_like(working_array_1)

    # initialise with prob = 1.0 in the zeroth position
    working_array_1[:, 0] = 1.0

    # Do dynamic-programming convolution along pmfs

    for i in range(n_items_per_dp_pmf):
        if i % 2:
            a_in, a_out = working_array_2, working_array_1
        else:
            a_in, a_out = working_array_1, working_array_2

        _DP_Kernel_2D(grid=(n_DP_rows, blocks_per_oa_pmf),
                      block=(threads_per_block,),
                      args=(a_in,
                            cp.array(p[i], dtype=np.float64),
                            cp.array(w[i], dtype=np.int32),
                            cp.array(n[i], dtype=np.int32),
                            a_out)
                      )

    a_out = a_out[:, :oa_pmf_length]

    if a_out.shape[0] == 1:
        return a_out[0]
    else:
        final_pmf_length = weights.sum() + 1
        a_out = _omnivolve_rows(a_out, final_pmf_length)
        return a_out


def _GPB_Fastest_CUDA(probs, weights):
    """Compute the pmf vector of the Generalised Poisson Binomial Distribution
    in two parameter form. Chooses the fastest method from the above

    Private Function without any type checking or input validation. Assumes
    probs and weights are the same length

    :param probs: (numpy.ndarray, 1D, dtype=numpy.float64): success probabilities
    :param weights: (numpy.ndarray, 1D, dtype=numpy.int32): positive success weights
    :return: (cupy.ndarray, 1D, dtype=numpy.float64) the pmf vector  
    """
    M = probs.shape[0]
    if M > 6000:
        method = _GPB_DP_DC_FFT_CUDA
    elif M < 500:
        method = _GPB_DP_CUDA
    elif log2(M) + log2(weights.sum() + 1) > 30:
        method = _GPB_DP_DC_FFT_CUDA
    else:
        method = _GPB_DP_CUDA
    return method(probs, weights)


_available_methods = {
    'DP_CUDA': _GPB_DP_CUDA,
    'DP_DC_FFT_CUDA': _GPB_DP_DC_FFT_CUDA,
    'Fastest_CUDA': _GPB_Fastest_CUDA
}

"""### Gauss Kronrod quadrature for use on GPU arrays"""

kronrod_nodes = np.array([-0.9914553711208126392069, -0.9491079123427585245262, -0.8648644233597690727897,
                          -0.7415311855993944398639, -0.5860872354676911302941, -0.4058451513773971669066,
                          -0.2077849550078984676007, 0.0000000000000000000000, 0.2077849550078984676007,
                          0.4058451513773971669066, 0.5860872354676911302941, 0.7415311855993944398639,
                          0.8648644233597690727897, 0.9491079123427585245262, 0.9914553711208126392069])

kronrod_weights = np.array([0.0229353220105292249637, 0.0630920926299785532907, 0.1047900103222501838399,
                            0.1406532597155259187450, 0.1690047266392679028266, 0.1903505780647854099133,
                            0.2044329400752988924142, 0.2094821410847278280130, 0.2044329400752988924142,
                            0.1903505780647854099130, 0.1690047266392679028266, 0.1406532597155259187452,
                            0.1047900103222501838400, 0.0630920926299785532907, 0.02293532201052922496373])

gauss_weights = np.array([0, 0.129484966168869693271, 0, 0.279705391489276667901,
                          0, 0.3818300505051189449504, 0, 0.4179591836734693877551,
                          0, 0.3818300505051189449504, 0, 0.2797053914892766679015,
                          0, 0.1294849661688696932706, 0])


def _quad_vec_GPU(func, a, b, epsilon=1e-9):
    """Adaptive Quadrature for a vector valued function which returns a cupy
    array on device rather than a numpy array on the host

    Private Function without any type checking or input validation. 

    :param func: (callable): function that takes a single numpy.float64 as a 
    paramenter and returns (cupy.ndarray, 1D, dtype=numpy.float64)
    :param a: (np.float64): lower bound for integration
    :param b: (np.float64): upper bound for integration    
    :param epsilon: (np.float64): the maximum error allowed
    :return: (cupy.ndarray, 1D, dtype=numpy.float64) the integral vector  
    """

    half_width = (b - a) / 2
    mid_point = (b + a) / 2

    z = half_width * kronrod_nodes[0] + mid_point
    point_estimate = func(z)
    k_integral = point_estimate * kronrod_weights[0]
    g_integral = point_estimate * gauss_weights[0]

    for node, k_weight, g_weight in zip(kronrod_nodes[1:], kronrod_weights[1:], gauss_weights[1:]):
        z = half_width * node + mid_point
        point_estimate = func(z)
        k_integral += point_estimate * k_weight
        g_integral += point_estimate * g_weight

    k_integral *= half_width
    g_integral *= half_width
    deviation = abs(k_integral - g_integral)

    error_estimate = abs(k_integral - g_integral).max()
    error_max = epsilon * (b - a)
    if error_estimate <= error_max:
        return k_integral
    else:
        return _quad_vec_GPU(func, a, mid_point, epsilon) + _quad_vec_GPU(func, mid_point, b, epsilon)
