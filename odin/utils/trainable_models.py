"""
Implementations of trainable ODE models, the classes that contain the theta
variables optimized during training.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Libraries
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf


class TrainableModelODE(ABC):
    """
    Abstract class of a trainable dynamical system. The parameters to be
    estimated are contained in the attribute self.theta as TensorFlow Variables
    and will be optimized during training.
    Implementing new model consists in overriding the two abstract functions
    below.
    """

    def __init__(self,
                 n_states: int,
                 n_points: int,
                 bounds: np.array = None):
        """
        Constructor.
        :param n_states: number of states in the system;
        :param n_points: number of observation points;
        :param bounds for the optimization of theta.
        """
        self.n_states = n_states
        self.n_points = n_points
        with tf.variable_scope('risk_main'):
            self._initialize_parameter_variables()
        self.n_params = tf.constant(self.theta.shape[0], dtype=tf.int32)
        self.theta = tf.reshape(self.theta, shape=[self.n_params, 1])
        if bounds is None:
            bounds = np.inf * np.ones([self.theta.shape[0], 2])
            bounds[:, 0] = - bounds[:, 0]
        if self.theta.shape[0] == 1:
            self.parameter_lower_bounds = np.asscalar(bounds[0, 0])
            self.parameter_upper_bounds = np.asscalar(bounds[0, 1])
        else:
            self.parameter_lower_bounds = np.reshape(bounds[:, 0],
                                                     self.theta.shape)
            self.parameter_upper_bounds = np.reshape(bounds[:, 1],
                                                     self.theta.shape)
        return

    @abstractmethod
    def _initialize_parameter_variables(self) -> None:
        """
        Abstract method to be implemented. Initialize the TensorFlow variables
        containing the parameters theta of the ODE system. This will be 1D
        vector called 'self.theta', tensorflow.Variable type.
        """
        self.theta = tf.Variable(0.0)
        return

    @abstractmethod
    def compute_gradients(self, x: tf.Tensor) -> tf.Tensor:
        """
        Abstract method to be implemented. Compute the gradients of the ODE,
        meaning f(X, self.theta).
        :param x: values of the time series observed, whose shape is
        [n_states, n_points].
        :return: TensorFlow Tensor containing the gradients, whose shape is
        [n_states, n_points].
        """
        return tf.constant()


class TrainableLotkaVolterraODE(TrainableModelODE):
    """
    Trainable Lotka-Volterra model.
    """

    def _initialize_parameter_variables(self) -> None:
        """
        Initialize the TensorFlow variables containing the parameters theta of
        the ODE system. This will be 1D vector called 'self.theta',
        tensorflow.Variable type.
        """
        self.theta = tf.Variable(tf.abs(tf.random_normal([4, 1],
                                                         mean=0.0,
                                                         stddev=1.0,
                                                         dtype=tf.float64)),
                                 name='theta',
                                 trainable=True)
        return

    def compute_gradients(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the gradients of the ODE, meaning f(X, self.theta).
        :param x: values of the time series observed, whose dimensions are
        [n_states, n_points].
        :return: TensorFlow Tensor containing the gradients, whose shape is
        [n_states, n_points].
        """
        grad1 = self.theta[0] * x[0:1, :]\
            - self.theta[1] * x[0:1, :] * x[1:2, :]
        grad2 = - self.theta[2] * x[1:2, :]\
            + self.theta[3] * x[0:1, :] * x[1:2, :]
        gradient_samples = tf.concat([grad1, grad2], 0)
        return gradient_samples


class TrainableFitzHughNagumoODE(TrainableModelODE):
    """
    Trainable FitzHug-Nagumo model.
    """

    def _initialize_parameter_variables(self) -> None:
        """
        Initialize the TensorFlow variables containing the parameters theta of
        the ODE system. This will be 1D vector called 'self.theta',
        tensorflow.Variable type.
        """
        self.theta = tf.Variable(tf.abs(tf.random_normal([3, 1],
                                                         mean=0.0,
                                                         stddev=1.0,
                                                         dtype=tf.float64)),
                                 name='theta',
                                 trainable=True)
        return

    def compute_gradients(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the gradients of the ODE, meaning f(X, self.theta).
        :param x: values of the time series observed, whose dimensions are
        [n_states, n_points].
        :return: TensorFlow Tensor containing the gradients, whose shape is
        [n_states, n_points].
        """
        grad1 = self.theta[2] * (x[0:1, :] - tf.pow(x[0:1, :], 3.0) / 3.0 +
                                 x[1:2, :])
        grad2 = - (x[0:1, :] - self.theta[0] + self.theta[1] * x[1:2, :])\
            / self.theta[2]
        gradient_samples = tf.concat([grad1, grad2], 0)
        return gradient_samples


class TrainableProteinTransductionODE(TrainableModelODE):
    """
    Trainable 5D Protein transduction model.
    """

    def _initialize_parameter_variables(self) -> None:
        """
        Initialize the TensorFlow variables containing the parameters theta of
        the ODE system. This will be 1D vector called 'self.theta',
        tensorflow.Variable type.
        """
        self.theta = tf.Variable(tf.abs(tf.random_normal([6, 1],
                                                         mean=0.0,
                                                         stddev=1.0,
                                                         dtype=tf.float64)),
                                 name='theta',
                                 trainable=True)
        return

    def compute_gradients(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the gradients of the ODE, meaning f(X, self.theta).
        :param x: values of the time series observed, whose dimensions are
        [n_states, n_points].
        :return: TensorFlow Tensor containing the gradients, whose shape is
        [n_states, n_points].
        """
        grad1 = - self.theta[0] * x[0:1, :]\
            - self.theta[1] * x[0:1, :] * x[2:3, :] + self.theta[2] * x[3:4, :]
        grad2 = self.theta[0] * x[0:1, :]
        grad3 = - self.theta[1] * x[0:1, :] * x[2:3, :]\
            + self.theta[2] * x[3:4, :]\
            + self.theta[4] * x[4:5, :] / (self.theta[5] + x[4:5, :])
        grad4 = self.theta[1] * x[0:1, :] * x[2:3, :]\
            - self.theta[2] * x[3:4, :] - self.theta[3] * x[3:4, :]
        grad5 = self.theta[3] * x[3:4, :]\
            - self.theta[4] * x[4:5, :] / (self.theta[5] + x[4:5, :])
        gradient_samples = tf.concat([grad1, grad2, grad3, grad4, grad5], 0)
        return gradient_samples


class TrainableLorenz96ODE(TrainableModelODE):
    """
    Trainable Lorenz '96 model.
    """

    def _initialize_parameter_variables(self) -> None:
        """
        Initialize the TensorFlow variables containing the parameters of the
        ODE system. This will be 1D vector called 'self.theta',
        tensorflow.Variable type.
        """
        self.theta = tf.Variable(tf.abs(tf.random_normal([1],
                                                         mean=0.0,
                                                         stddev=1.0,
                                                         dtype=tf.float64)),
                                 name='theta',
                                 trainable=True)
        self.theta = tf.reshape(self.theta, [1, 1])
        return

    def compute_gradients(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the gradients of the ODE, meaning f(X, self.theta).
        :param x: values of the time series observed, whose dimensions are
        [n_states, n_points].
        :return: TensorFlow Tensor containing the gradients, whose shape is
        [n_states, n_points].
        """
        grad1 = (x[1:2, :] - x[self.n_states - 2:self.n_states - 1, :])\
            * x[self.n_states - 1:self.n_states, :] - x[0:1, :] + self.theta
        grad2 = (x[2:3, :] - x[self.n_states - 1:self.n_states, :])\
            * x[0:1, :] - x[1:2, :] + self.theta
        grad_list = [grad1, grad2]
        for n in range(2, self.n_states - 1):
            state_derivative = (x[n + 1:n + 2, :] - x[n - 2:n - 1, :])\
                * x[n - 1:n, :] - x[n:n + 1, :] + self.theta
            grad_list.append(state_derivative)
        state_derivative = \
            (x[0:1, :] - x[self.n_states - 3:self.n_states - 2, :]) \
            * x[self.n_states - 2:self.n_states - 1, :]\
            - x[self.n_states - 1:self.n_states, :] + self.theta
        grad_list.append(state_derivative)
        gradients = tf.concat(grad_list, axis=0)
        return gradients
