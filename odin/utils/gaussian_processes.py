"""
Gaussian Process class.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Libraries
from odin.utils.kernels import *
import tensorflow as tf
import numpy as np
import sys


class GaussianProcess(object):
    """
    Gaussian Process class for regression. The whole pipeline runs in
    TensorFlow in such a way that it's automatically differentiable.
    """

    def __init__(self,
                 input_dim: int,
                 n_points: int,
                 kernel: str = 'RBF',
                 use_single_gp: bool = False):
        """
        Constructor.
        :param input_dim: number of states;
        :param n_points: number of observation points;
        :param kernel: string indicating which kernel to use for regression.
        Valid options are 'RBF', 'Matern52', 'Matern32', 'RationalQuadratic',
        'Sigmoid';
        :param use_single_gp: boolean, indicates whether to use a single set of
        hyperparameters for all states (useful for extremely scarce data
        setting).
        """
        self.n_states = tf.constant(input_dim, dtype=tf.int32)
        self.n_points = tf.constant(n_points, dtype=tf.int32)
        self.jitter = tf.constant(1e-4, dtype=tf.float64)
        self.kernel = self._initialize_kernel(input_dim,
                                              kernel,
                                              use_single_gp)
        self._initialize_variables(use_single_gp)
        # GP Regression matrices
        self.c_phi_matrices = None
        self.c_phi_matrices_noiseless = None
        self.cross_c_phi_matrices = None
        self.diff_c_phi_matrices = None
        self.c_phi_diff_matrices = None
        self.diff_c_phi_diff_matrices = None
        return

    @staticmethod
    def _initialize_kernel(input_dim: int,
                           kernel: str = 'RBF',
                           use_single_gp: bool = False)\
            -> GenericKernel:
        """
        Initialize the kernel of the Gaussian Process.
        :param input_dim: number of states;
        :param kernel: string indicating which kernel to use for regression.
        Valid options are 'RBF', 'Matern52', 'Matern32', 'RationalQuadratic',
        'Sigmoid';
        :param use_single_gp: boolean, indicates whether to use a single set of
        hyperparameters for all states (useful for extremely scarce data
        setting).
        :return: the GenericKernel object.
        """
        if kernel == 'RBF':
            return RBFKernel(input_dim, use_single_gp)
        elif kernel == 'Matern52':
            return Matern52Kernel(input_dim, use_single_gp)
        elif kernel == 'Matern32':
            return Matern32Kernel(input_dim, use_single_gp)
        elif kernel == 'RationalQuadratic':
            return RationalQuadraticKernel(input_dim,
                                           use_single_gp)
        elif kernel == 'Sigmoid':
            return SigmoidKernel(input_dim, use_single_gp)
        else:
            sys.exit("Error: specified Gaussian Process kernel not valid")

    def _initialize_variables(self, use_single_gp: bool) -> None:
        """
        Initialize the variance of the log-likelihood of the GP as a TensorFlow
        variable. A logarithm-exponential transformation is used to ensure
        positivity during optimization.
        :param use_single_gp: boolean, indicates whether to use a single set of
        hyperparameters for all states (useful for extremely scarce data
        setting).
        """
        with tf.variable_scope('gaussian_process'):
            if use_single_gp:
                self.likelihood_logvariance = tf.Variable(
                    np.log(1.0), dtype=tf.float64, trainable=True,
                    name='variance_loglik')
                self.likelihood_logvariances =\
                    self.likelihood_logvariance * tf.ones([self.n_states,
                                                           1, 1],
                                                          dtype=tf.float64)
            else:
                self.likelihood_logvariances = tf.Variable(
                    np.log(1.0) * tf.ones([self.n_states, 1, 1],
                                          dtype=tf.float64),
                    dtype=tf.float64, trainable=True,
                    name='variances_loglik')
            self.likelihood_variances = tf.exp(self.likelihood_logvariances)
        return

    def build_supporting_covariance_matrices(self, t: tf.Tensor,
                                             t_new: tf.Tensor) -> None:
        """
        Pre-compute the GP matrices as TensorFlow tensors.
        :param t: time stamps of the training set;
        :param t_new: time stamps of the test points (usually same as before);
        """
        self.c_phi_matrices =\
            self._build_c_phi_matrices(t)
        self.c_phi_matrices_noiseless =\
            self._build_c_phi_matrices_noiseless(t)
        self.cross_c_phi_matrices =\
            self._build_cross_c_phi_matrices(t, t_new)
        self.diff_c_phi_matrices = \
            self._build_diff_c_phi_matrices(t, t_new)
        self.c_phi_diff_matrices = \
            self._build_c_phi_diff_matrices(t, t_new)
        self.diff_c_phi_diff_matrices =\
            self._build_diff_c_phi_diff_matrices(t_new)
        return

    def _build_c_phi_matrices(self, t: tf.Tensor) -> tf.Tensor:
        """
        Build the covariance matrices K(x_train, x_train) + sigma_y^2 I.
        :param t: time stamps of the training set;
        :return: the tensors containing the matrices.
        """
        c_phi_matrices = self.kernel.compute_c_phi(t, t)\
            + tf.expand_dims(tf.eye(self.n_points, dtype=tf.float64), 0)\
            * self.likelihood_variances
        return c_phi_matrices

    def _build_c_phi_matrices_noiseless(self, t: tf.Tensor) -> tf.Tensor:
        """
        Build the covariance matrices K(x_train, x_train).
        :param t: time stamps of the training set;
        :return: the tensors containing the matrices.
        """
        c_phi_matrices = self.kernel.compute_c_phi(t, t)\
            + tf.expand_dims(tf.eye(self.n_points, dtype=tf.float64), 0)\
            * self.jitter
        return c_phi_matrices

    def _build_cross_c_phi_matrices(self,
                                    t: tf.Tensor,
                                    t_new: tf.Tensor) -> tf.Tensor:
        """
        Build the cross covariance matrices between the training data and the
        new test points: K(x_train, x_test).
        :param t: time stamps of the training set;
        :param t_new: time stamps of the test points (usually same as before);
        :return: the tensors containing the matrices.
        """
        cross_c_phi_matrices = self.kernel.compute_c_phi(t, t_new)
        return cross_c_phi_matrices

    def _build_diff_c_phi_matrices(self, t: tf.Tensor,
                                   t_new: tf.Tensor)\
            -> tf.Tensor:
        """
        Builds the matrices diff_c_phi: dK(t,t') / dt.
        :param t: time stamps of the training set;
        :param t_new: time stamps of the test points (usually same as before);
        :return the tensor containing the matrices.
        """
        diff_c_phi_matrices = self.kernel.compute_diff_c_phi(t_new, t)
        return diff_c_phi_matrices

    def _build_c_phi_diff_matrices(self, t: tf.Tensor,
                                   t_new: tf.Tensor)\
            -> tf.Tensor:
        """
        Builds the matrices c_phi_diff: dK(t,t') / dt'.
        :param t: time stamps of the training set;
        :param t_new: time stamps of the test points (usually same as before);
        :return the tensor containing the matrices.
        """
        c_phi_diff_matrices = self.kernel.compute_c_phi_diff(t, t_new)
        return c_phi_diff_matrices

    def _build_diff_c_phi_diff_matrices(self, t_new: tf.Tensor)\
            -> tf.Tensor:
        """
        Builds the matrices diff_c_phi_diff: d^2K(t,t') / dt dt'.
        :param t_new: time stamps of the test points (usually same as before);
        :return the tensor containing the matrices.
        """
        diff_c_phi_diff_matrices = self.kernel.compute_diff_c_phi_diff(t_new,
                                                                       t_new)
        return diff_c_phi_diff_matrices

    def compute_posterior_mean(self, system: tf.Tensor) -> tf.Tensor:
        """
        Compute the mean of GP the posterior.
        :param system: values of the states of the system;
        :return: the TensorFlow tensor with the mean.
        """
        y_matrix = tf.linalg.solve(self.c_phi_matrices,
                                   tf.expand_dims(system, -1))
        mu = tf.matmul(self.cross_c_phi_matrices, y_matrix)
        return mu

    def compute_posterior_variance(self) -> tf.Tensor:
        """
        Compute the posterior variance matrix of the training points.
        :return: the TensorFlow tensor with the variance matrix.
        """
        a_matrix = tf.linalg.solve(self.c_phi_matrices,
                                   self.cross_c_phi_matrices)
        fvar = self.c_phi_matrices_noiseless\
            - tf.matmul(self.cross_c_phi_matrices, a_matrix, transpose_a=True)
        return fvar

    def compute_posterior_derivative_mean(self, system: tf.Tensor) -> tf.Tensor:
        """
        Compute the mean of derivative GP posterior.
        :param system: values of the states of the system;
        :return: the TensorFlow tensor with the mean.
        """
        v_matrix = tf.linalg.solve(self.c_phi_matrices_noiseless,
                                   tf.expand_dims(system, -1))
        mu = tf.matmul(self.diff_c_phi_matrices,
                       v_matrix)
        return mu

    def compute_posterior_derivative_variance(self) -> tf.Tensor:
        """
        Compute the derivative posterior variance matrix of the training points.
        :return: the TensorFlow tensor with the variance matrix.
        """
        second_term = tf.matmul(
            self.diff_c_phi_matrices,
            tf.linalg.solve(self.c_phi_matrices_noiseless,
                            self.c_phi_diff_matrices))
        fvar = self.diff_c_phi_diff_matrices - second_term\
            + tf.expand_dims(tf.eye(self.n_points, dtype=tf.float64), 0) \
            * self.jitter
        return fvar

    def compute_average_log_likelihood(self, system: tf.Tensor) -> tf.Tensor:
        """
        Compute the log-likelihood of the data passed as argument.
        :param system: values of the states of the system;
        :return: the tensor containing the log-likelihood.
        """
        logdets_cov_matrices = tf.linalg.logdet(self.c_phi_matrices)
        y_matrix = tf.linalg.solve(self.c_phi_matrices,
                                   tf.expand_dims(system, -1))
        first_term = tf.reduce_sum(system * tf.squeeze(y_matrix), axis=1)
        log_likelihood = -0.5 * (first_term + logdets_cov_matrices)
        return tf.reduce_sum(log_likelihood / tf.cast(self.n_points,
                                                      tf.float64))
