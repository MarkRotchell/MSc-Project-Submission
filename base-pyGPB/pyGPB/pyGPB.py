import numpy as np
import subprocess
import inspect

from math import sqrt, log2

from scipy.stats._distn_infrastructure import rv_sample
from scipy.integrate import quad_vec
from scipy.stats import norm
from scipy.special import ndtri as norm_ppf

import pyGPB._cpu_methods as cpu

# Import GPU functionality if there is a CUDA gpu available
try:
    subprocess.check_output('nvidia-smi')
    import pyGPB._gpu_methods as gpu

    GPU_AVAILABLE = True

except Exception:
    GPU_AVAILABLE = False


class GPB(rv_sample):
    """A Generalised Poisson Binomial random variable.

    Distribution paramters can be supplied in either 2-parameter (p, w) or 3-parameter (p, w_s, w_f) form:

    **2-Parameter Form**:

    Requires probs and weights. Weights should be non-zero. An event failure is assumed to have zero weight.

    **3-Parameter Form**:

    Requires probs, weights and failure_weights. Success and Failure weights for each event must be different.

    :param probs: Success probabilities. Must all be between 0 and 1 (exclusive)
    :type probs: 1D :class:`numpy.ndarray` (or other array-like) of float
    :param weights: Success weights. If failure_weights is not provided then must all be non-zero
    :type weights: 1D :class:`numpy.ndarray` (or other array-like) of int
    :param failure_weights: Failure weights. If provided then must be different to weights (elementwise)
    :type failure_weights: 1D :class:`numpy.ndarray` (or other array-like) of int, optional
    :param method: The function used to compute the pmf vector. If ``None`` is provided, then the appropriate function will
        be chosen depending on the other parameters. If a callable is provided then it will be assumed to be able to
        compute the pmf vector of the GPB with parameters (probs, weights). If string is passed then the function is
        found by looking up in each sub-module's :py:data:`_available_methods` dict.

        | Currently Available GPU Method strings:

        * 'DP_CUDA': :py:func:`_GPB_DP_CUDA`
        * 'DP_DC_FFT_CUDA': :py:func:`_GPB_DP_DC_FFT_CUDA`
        * 'Fastest_CUDA': :py:func:`_GPB_Fastest_CUDA`

        | Currently Available CPU Method strings:

        * 'Naive': :py:func:`_GPB_Naive`
        * 'CF': :py:func:`_GPB_CF`
        * 'DP': :py:func:`_GPB_DP`
        * 'FFT': :py:func:`_GPB_DC_FFT`
        * 'Fastest_CPU': :py:func:`_GPB_DP_DC_FFT_Combo`
    :type method: string, callable or ``None``, optional
    :param allow_GPU: Whether pyGPB should allow GPU methods if available.
    :type allow_GPU: bool, optional
    :param prefer_speed: Whether to prioritise the faster FFT methods, accepting the relative-error issues in the tail.
        If false then will always use dynamic-programming methods which may be slower for larger N and M
    :type prefer_speed: bool, optional

    """

    def __new__(cls, probs, weights, failure_weights=None, method=None,
                allow_GPU=True, prefer_speed=True):
        return super(GPB, cls).__new__(cls)

    def __init__(self, probs, weights, failure_weights=None, method=None,
                 allow_GPU=True, prefer_speed=True):

        # Set probs and weights
        self._set_probs_and_weights(probs, weights, failure_weights)

        # Choose appropriate method for computing the pmf vector
        self._set_method(method, allow_GPU, prefer_speed)

        # Calculate the pmf vector
        self._pmf_vec = self.method(probs=self._probs_for_calc, weights=self._weights_for_calc)

        # If result is a GPU device array then move it to host
        if self.used_gpu and not isinstance(self._pmf_vec, np.ndarray):
            self._pmf_vec = self._pmf_vec.get()

        # Ensure all values are positive (not guaranteed for FFT approaches)
        self._pmf_vec = np.where(self._pmf_vec < 0, 0, self._pmf_vec)

        # Calculate cdf vector
        self._cdf_vec = self._pmf_vec.cumsum()

        # Pass remaning instantiation to parent class, rv_sample which is a 
        # subclass of rv_discrete which allows for construction by passing
        # support and pmf

        super().__init__(
            self.support_start, b=self.support_end, inc=1,
            values=(np.arange(self.support_start, self.support_end + 1), self._pmf_vec))

    def _set_probs_and_weights(self, probs, weights, failure_weights):
        """Check distribution parameters for correctness and add them to object state
        if 3-param version is used, or 2-param with negative weights then need to 
        re-arrange slightly a form with only positive weights. These params are
        then stored in self._probs_for_calc, and self._weights_for_calc

    :param probs: Success probabilities. Must all be between 0 and 1 (exclusive)
    :type probs: 1D :class:`numpy.ndarray` (or other array-like) of float
    :param weights: Success weights. If failure_weights is not provided then must all be non-zero
    :type weights: 1D :class:`numpy.ndarray` (or other array-like) of int
    :param failure_weights: Failure weights. If provided then must be different to weights (elementwise)
    :type failure_weights: 1D :class:`numpy.ndarray` (or other array-like) of int, optional
        """
        # Check probabilities for correctness

        try:
            self.probs = np.array(probs, dtype=np.float64)
        except Exception:
            raise ValueError('probs should be a numpy float64 array or castable to a numpy float64')

        if self.probs.ndim != 1:
            raise ValueError('probs should be one dimensional')

        if np.any(self.probs <= 0) or np.any(self.probs >= 1):
            raise ValueError('probs should be between zero and one exclusive')

        # Check Weights for Correctness
        try:
            self.weights = np.array(weights, dtype=np.int32)
        except Exception:
            raise ValueError('weights should be a numpy int32 array or castable to a numpy int32')

        if self.weights.ndim != 1:
            raise ValueError('weights should be one dimensional')

        # Check weights and probs are the same length
        if self.probs.shape[0] != self.weights.shape[0]:
            raise ValueError('probs and weights should be the same length')

        # Check failure weights

        if failure_weights is not None:
            try:
                self.failure_weights = np.array(failure_weights, dtype=np.int32)
            except Exception:
                raise ValueError('failure_weights should be a numpy int32 array or castable to a numpy int32')

            if self.failure_weights.ndim != 1:
                raise ValueError('failure_weights should be one dimensional')

            if self.failure_weights.shape[0] != self.weights.shape[0]:
                raise ValueError('failure_weights and weights should be the same length')

            if np.any(self.failure_weights == self.weights):
                raise ValueError('failure_weights and weights should be different for all values')

        elif np.any(self.weights == 0):
            raise ValueError('success weights should be none zero or have a non-zero failure weight (but not both)')

        else:
            self.failure_weights = np.zeros_like(self.weights, dtype=np.int32)

        # Re-arrange so that only postive weights are presented to the calculation method. 
        # The if failure_weight is lower than success_weight then they are swapped the prob is flipped to 1-prob
        # They have the lower of the two subtracted so that the failure weight is zero, with the total difference 
        # Kept as an offset to apply to the support

        self._probs_for_calc = np.where(self.weights > self.failure_weights, self.probs, 1 - self.probs)
        self._weights_for_calc = np.maximum(self.weights, self.failure_weights) - np.minimum(self.weights,
                                                                                             self.failure_weights)
        self.support_start = np.minimum(self.weights, self.failure_weights).sum()
        self.support_end = self.support_start + self._weights_for_calc.sum()

    def _set_method(self, method, allow_GPU=True, prefer_speed=True):
        """Check method parameters for correctness and add them to object state
        If necessary use heuristic for choosing best method

        :method: (string, callable or None): The function used to compute the pmf
            vector. If None is provided, then the appropriate function will be 
            chosen depending on the other parameters. If a callable is provided 
            then it will be assumed to be able to compute the pmf vector of the GPB
            with parameters (probs, weights). If string is passed then the function 
            is found by looking up in each sub-module's _available_methods dict.
        :allow_GPU: (boolean): whether pyGPB should allow GPU methods if available.
        :prefer_speed: (boolean): whether to priorities the faster FFT methods, 
            accepting the relative-error issues in the tail. If false then will 
            always use dynamic-programming methods which may be slower for larger
            N and M  
        """
        # check allow_GPU
        try:
            self.allow_GPU = bool(allow_GPU)
            self._can_use_GPU = self.allow_GPU and GPU_AVAILABLE
        except Exception:
            raise ValueError('allow_GPU should be bool or castable to bool')

        # check prefer_speed
        try:
            self.prefer_speed = bool(prefer_speed)
        except Exception:
            raise ValueError('prefer_speed should be bool or castable to bool')

            # Choose appropriate method
        if callable(method):
            # User has provided own method
            self.method = method
            params = inspect.signature(self.method).parameters
            if not ('probs' in params and 'weights' in params):
                raise ValueError('Any passed method should have "probs" and "weights" as named arguments')
            self.used_gpu = self.method in gpu._available_methods
        elif isinstance(method, str):
            # User has provided a string describing the method they wish to use
            if method in cpu._available_methods:
                self.method = cpu._available_methods[method]
                self.used_gpu = False
            elif self._can_use_GPU and method in gpu._available_methods:
                self.method = gpu._available_methods[method]
                self.used_gpu = True
            else:
                raise ValueError(f'method "{method}" not recognised')
        elif method is None:
            # User has not specified a method. Choose for them:
            if self._can_use_GPU:
                M = self._weights_for_calc.shape[0]
                N = self._weights_for_calc.sum() + 1
                # Heuristic test for when GPU is faster
                # TODO: improve this test for general hardware
                self.used_gpu = (log2(M) + log2(N) > 19) and ((M > 1200) or (N > 6000))
            else:
                self.used_gpu = False
            if self.used_gpu and self.prefer_speed:
                self.method = gpu._GPB_Fastest_CUDA
            elif self.used_gpu:
                self.method = gpu._GPB_DP_CUDA
            elif self.prefer_speed:
                self.method = cpu._GPB_DP_DC_FFT_Combo
            else:
                self.method = cpu._GPB_DP
        else:
            raise ValueError(f'method of type"{type(method)}" not recognised')

    @property
    def pmf_vec(self):
        """The full Probability Mass vector.

        :returns: the pmf vector.
        :rtype: :class:`numpy.ndarray` (or array like). 1D, dtype= :class:`numpy.float64`
        """
        return self._pmf_vec

    @property
    def cdf_vec(self):
        """The Cumulative Distribution Function in vector form.

        :returns: the cdf vector.
        :rtype: :class:`numpy.ndarray` (or array like). 1D, dtype= :class:`numpy.float64`
        """
        return self._cdf_vec


class LFGPB(GPB):
    """A Latent Factor Generalised Poisson Binomial random variable.

    Distribution paramters can be supplied in either 3-parameter (p, w, rho) or 4-parameter (p, w_s, rho, w_f) form:

    ## 3-Parameter Form:

    Requires probs and weights and rho. Weights should be non-zero. Event Failure is assumed to be weight 0.

    ## 4-Parameter Form:

    Requires probs, weights, failure weights and rho. Failure and Success weight for each event must be different.


    Requires probs, weights and failure_weights. Success and Failure weights for each event must be different.

    :param probs: Success probabilities. Must all be between 0 and 1 (exclusive)
    :type probs: 1D :class:`numpy.ndarray` (or other array-like) of float
    :param weights: Success weights. If failure_weights is not provided then must all be non-zero
    :type weights: 1D :class:`numpy.ndarray` (or other array-like) of int
    :param failure_weights: Failure weights. If provided then must be different to weights (elementwise)
    :type failure_weights: 1D :class:`numpy.ndarray` (or other array-like) of int, optional
    :param method: The function used to compute the pmf vector. If ``None`` is provided, then the appropriate function will
        be chosen depending on the other parameters. If a callable is provided then it will be assumed to be able to
        compute the pmf vector of the GPB with parameters (probs, weights). If string is passed then the function is
        found by looking up in each sub-module's :py:data:`_available_methods` dict.

        | Currently Available GPU Method strings:

        * 'DP_CUDA': :py:func:`_GPB_DP_CUDA`
        * 'DP_DC_FFT_CUDA': :py:func:`_GPB_DP_DC_FFT_CUDA`
        * 'Fastest_CUDA': :py:func:`_GPB_Fastest_CUDA`

        | Currently Available CPU Method strings:

        * 'Naive': :py:func:`_GPB_Naive`
        * 'CF': :py:func:`_GPB_CF`
        * 'DP': :py:func:`_GPB_DP`
        * 'FFT': :py:func:`_GPB_DC_FFT`
        * 'Fastest_CPU': :py:func:`_GPB_DP_DC_FFT_Combo`
    :type method: string, callable or ``None``, optional
    :param allow_GPU: Whether pyGPB should allow GPU methods if available.
    :type allow_GPU: bool, optional
    :param prefer_speed: Whether to prioritise the faster FFT methods, accepting the relative-error issues in the tail.
        If false then will always use dynamic-programming methods which may be slower for larger N and M
    :type prefer_speed: bool, optional
    :param rho: the constant pairwise correlation for event success. Default value is 0, i.e. mutual independence.
    :type rho: float, optional
    :param epsilon: allowable max error for numerical integration.
    :type epsilon: float, optional

    """


    def __new__(cls, probs, weights, rho=None, failure_weights=None, method=None, allow_GPU=True, prefer_speed=True,
                epsilon=1e-9):
        return super(GPB, cls).__new__(cls)

    def __init__(self, probs, weights, rho=None, failure_weights=None, method=None, allow_GPU=True, prefer_speed=True,
                 epsilon=1e-9):
        """
        # Initialisation
        
        """

        if rho is None or rho == 0:
            super().__init__(probs, weights, failure_weights, method,
                             allow_GPU, prefer_speed)
        else:
            try:
                self.rho = float(rho)
            except Exception:
                raise ValueError('rho must be None or castable to float')

            if not 0 < rho < 1:
                raise ValueError('rho must be positive and less than one')

            try:
                self.epsilon = float(epsilon)
            except Exception:
                raise ValueError('epsilon must be castable to float')

            if not 0 < epsilon < 1:
                raise ValueError('epsilon must be positive and less than one')

            # Set probs and weights. Parent method is used
            self._set_probs_and_weights(probs, weights, failure_weights)

            # Choose appropriate method for computing the pmf vector. Parent
            # method is used
            self._set_method(method, allow_GPU, prefer_speed)

            # Calculate the probability distribution vectors
            self._compute_distributions()

            # Pass remaning instantiation to parent class, rv_sample which is a 
            # subclass of rv_discrete which allows for construction by passing
            # support and pmf

            rv_sample.__init__(self,
                               self.support_start, b=self.support_end, inc=1,
                               values=(np.arange(self.support_start, self.support_end + 1), self._pmf_vec))

    def _compute_distributions(self):
        # Compute the pmf (and cdf) vectors
        probs_ppf = norm.ppf(self.probs)
        scale = sqrt(1 - self.rho)
        root_rho = sqrt(self.rho)

        def cond_gpb(z):
            # Partial Function: computes the GPB conditional on z
            cond_probs = norm.cdf(probs_ppf, loc=root_rho * norm_ppf(z), scale=scale)
            cond_probs = np.where(self.failure_weights > self.weights, 1 - cond_probs, cond_probs)
            return self.method(cond_probs, self._weights_for_calc)

        # Perform quadrature
        if self.used_gpu:
            self._pmf_vec = gpu._quad_vec_GPU(cond_gpb, 0, 1, self.epsilon).get()
        else:
            self._pmf_vec = quad_vec(cond_gpb, 0, 1, self.epsilon, self.epsilon, norm='max')[0]

        # Ensure all elements are positive
        self._pmf_vec = np.where(self._pmf_vec < 0, 0, self._pmf_vec)

        # Compute the cdf
        self._cdf_vec = self._pmf_vec.cumsum()
