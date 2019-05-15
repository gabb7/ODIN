"""
Main implementation of the ODIN algorithm.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Libraries
from odin.utils.trainable_models import TrainableModel
from odin.utils.gaussian_processes import GaussianProcess
from odin.utils.tensorflow_optimizer import ExtendedScipyOptimizerInterface
import numpy as np
import tensorflow as tf
from typing import Union, Tuple


class ODIN(object):
    """
    Class that implements the main ODIN regression algorithm.
    """

    def __init__(self,
                 trainable: TrainableModel,
                 system_data: np.array,
                 t_data: np.array,
                 gp_kernel: str = 'RBF',
                 optimizer: str = 'L-BFGS-B',
                 initial_gamma: float = 0.3,
                 train_gamma: bool = True,
                 gamma_bounds: Union[np.array, list, Tuple] = (1e-6, 10.0),
                 state_bounds: np.array = None,
                 basinhopping: bool = True,
                 basinhopping_options: dict = None,
                 single_gp: bool = False,
                 state_normalization: bool = True,
                 time_normalization: bool = True):
        """
        Constructor.
        :param trainable: Trainable model class, as explained and implemented in
        utils.trainable_models;
        :param system_data: numpy array containing the noisy observations of
        the state values of the system, size is [n_states, n_points];
        :param t_data: numpy array containing the time stamps corresponding to
        the observations passed as system_data;
        :param gp_kernel: string indicating which kernel to use in the GP.
        Valid options are 'RBF', 'Matern52', 'Matern32', 'RationalQuadratic',
        'Sigmoid';
        :param optimizer: string indicating which scipy optimizer to use. The
        valid ones are the same that can be passed to scipy.optimize.minimize.
        Notice that some of them will ignore bounds;
        :param initial_gamma: initial value for the gamma parameter.
        :param train_gamma: boolean, indicates whether to train of not the
        variable gamma;
        :param gamma_bounds: bounds for gamma (a lower bound of at least 1e-6
        is always applied to overcome numerical instabilities);
        :param state_bounds: bounds for the state optimization;
        :param basinhopping: boolean, indicates whether to turn on the scipy
        basinhopping;
        :param basinhopping_options: dictionary containing options for the
        basinhooping algorithm (syntax is the same as scipy's one);
        :param single_gp: boolean, indicates whether to use a single set of GP
        hyperparameters for each state;
        :param state_normalization: boolean, indicates whether to normalize the
        states values before the optimization (notice the parameter values
        theta won't change);
        :param time_normalization: boolean, indicates whether to normalize the
        time stamps before the optimization (notice the parameter values
        theta won't change).
        """
        # Save arguments
        self.trainable = trainable
        self.system_data = np.copy(system_data)
        self.t_data = np.copy(t_data).reshape(-1, 1)
        self.n_states, self.n_p = system_data.shape
        self.gp_kernel = gp_kernel
        self.optimizer = optimizer
        self.initial_gamma = initial_gamma
        self.train_gamma = train_gamma
        self.basinhopping = basinhopping
        self.basinhopping_options = {'n_iter': 10,
                                     'temperature': 1.0,
                                     'stepsize': 0.05}
        if basinhopping_options:
            self.basinhopping_options.update(basinhopping_options)
        self.single_gp = single_gp
        # Build bounds for the states and for gamma
        self._compute_state_bounds(state_bounds)
        self._compute_gamma_bounds(gamma_bounds)
        # Compute the data for the standardization (means and standard
        # deviations)
        self._compute_standardization_data(state_normalization,
                                           time_normalization)
        # Build the necessary TensorFlow tensors
        self._build_tf_data()
        # Initialize the Gaussian Process for the derivative model
        self.gaussian_process = GaussianProcess(self.n_states, self.n_p,
                                                self.gp_kernel, self.single_gp)
        # Initialization of TF operations
        self.init = None
        self.model_gp_loglikelihood = None
        self.negative_data_loglikelihood = None
        return

    def _compute_state_bounds(self, bounds: np.array) -> None:
        """
        Builds the numpy array that defines the bounds for the states.
        :param bounds: numpy array, sized [n_dim, 2], in which for each
        dimensions we can find respectively lower and upper bounds.
        """
        if bounds is None:
            self.state_bounds = np.inf * np.ones([self.n_states, 2])
            self.state_bounds[:, 0] = - self.state_bounds[:, 0]
        else:
            self.state_bounds = np.array(bounds)
        return

    def _compute_gamma_bounds(self, bounds: Union[np.array, list, Tuple])\
            -> None:
        """
        Builds the numpy array that defines the bounds for gamma.
        :param bounds: of the form (lower_bound, upper_bound).
        """
        self.gamma_bounds = np.array([1.0, 1.0])
        if bounds is None:
            self.gamma_bounds[0] = np.log(1e-6)
            self.gamma_bounds[1] = np.inf
        else:
            self.gamma_bounds[0] = np.log(np.array(bounds[0]))
            self.gamma_bounds[1] = np.log(np.array(bounds[1]))
        return

    def _compute_standardization_data(self, state_normalization: bool,
                                      time_normalization: bool) -> None:
        """
        Compute the means and the standard deviations for data standardization,
        used in the GP regression.
        """
        # Compute mean and std dev of the state and time values
        if state_normalization:
            self.system_data_means = np.mean(self.system_data,
                                             axis=1).reshape(self.n_states, 1)
            self.system_data_std_dev = np.std(self.system_data,
                                              axis=1).reshape(self.n_states, 1)
        else:
            self.system_data_means = np.zeros([self.n_states, 1])
            self.system_data_std_dev = np.ones([self.n_states, 1])
        if time_normalization:
            self.t_data_mean = np.mean(self.t_data)
            self.t_data_std_dev = np.std(self.t_data)
        else:
            self.t_data_mean = 0.0
            self.t_data_std_dev = 1.0
        # For the sigmoid kernel the input time values must be positive, i.e.
        # we only divide by the standard deviation
        if self.gp_kernel == 'Sigmoid':
            self.t_data_mean = 0.0
        # Normalize states and time
        self.normalized_states = (self.system_data - self.system_data_means) / \
            self.system_data_std_dev
        self.normalized_t_data = (self.t_data - self.t_data_mean) / \
            self.t_data_std_dev
        return

    def _build_tf_data(self) -> None:
        """
        Initialize all the TensorFlow constants needed in the pipeline.
        """
        self.system = tf.constant(self.normalized_states, dtype=tf.float64)
        self.t = tf.constant(self.normalized_t_data, dtype=tf.float64)
        self.system_means = tf.constant(self.system_data_means,
                                        dtype=tf.float64,
                                        shape=[self.n_states, 1])
        self.system_std_dev = tf.constant(self.system_data_std_dev,
                                          dtype=tf.float64,
                                          shape=[self.n_states, 1])
        self.t_mean = tf.constant(self.t_data_mean, dtype=tf.float64)
        self.t_std_dev = tf.constant(self.t_data_std_dev, dtype=tf.float64)
        self.n_points = tf.constant(self.n_p, dtype=tf.int32)
        self.dimensionality = tf.constant(self.n_states, dtype=tf.int32)
        return

    @staticmethod
    def _build_var_to_bounds_gp() -> dict:
        """
        Builds the dictionary containing the bounds that will be applied to the
        variable in the Gaussian Process model.
        :return: the dictionary variables to bounds.
        """
        # Extract TF variables and select the GP ones
        t_vars = tf.trainable_variables()
        gp_vars = [var for var in t_vars if 'gaussian_process' in var.name]
        # Bounds for the GP hyper-parameters
        gp_kern_lengthscale_bounds = (np.log(1e-6), np.log(100.0))
        gp_kern_variance_bounds = (np.log(1e-6), np.log(100.0))
        gp_kern_likelihood_bounds = (np.log(1e-6), np.log(100.0))
        # Dictionary construction
        var_to_bounds = {gp_vars[0]: gp_kern_lengthscale_bounds,
                         gp_vars[1]: gp_kern_variance_bounds,
                         gp_vars[2]: gp_kern_likelihood_bounds}
        return var_to_bounds

    @staticmethod
    def _build_var_to_bounds_gp_sigmoid() -> dict:
        """
        Builds the dictionary containing the bounds that will be applied to the
        variable in the Gaussian Process model (specific for the sigmoid
        kernel).
        :return: the dictionary variables to bounds.
        """
        # Extract TF variables and select the GP ones
        t_vars = tf.trainable_variables()
        gp_vars = [var for var in t_vars if 'gaussian_process' in var.name]
        # Bounds for the GP hyper-parameters
        gp_kern_a_bounds = (np.log(1e-6), np.log(100.0))
        gp_kern_b_bounds = (np.log(1e-6), np.log(100.0))
        gp_kern_variance_bounds = (np.log(1e-6), np.log(100.0))
        gp_kern_likelihood_bounds = (np.log(1e-6), np.log(100.0))
        # Dictionary construction
        var_to_bounds = {gp_vars[0]: gp_kern_a_bounds,
                         gp_vars[1]: gp_kern_b_bounds,
                         gp_vars[2]: gp_kern_variance_bounds,
                         gp_vars[3]: gp_kern_likelihood_bounds}
        return var_to_bounds

    def _train_data_based_gp(self, session: tf.Session()) -> None:
        """
        Performs a classic GP regression on the data of the system. For each
        state of the system we train a different GP by maximum likelihood to fix
        the kernel hyper-parameters.
        :param session: TensorFlow session used during the optimization.
        """
        # Extract TF variables and select the GP ones
        t_vars = tf.trainable_variables()
        gp_vars = [var for var in t_vars if 'gaussian_process' in var.name]
        # Build the bounds for the GP hyper-parameters
        if self.gp_kernel == 'Sigmoid':
            var_to_bounds = self._build_var_to_bounds_gp_sigmoid()
        else:
            var_to_bounds = self._build_var_to_bounds_gp()
        # Initialize the TF/scipy optimizer
        self.data_gp_optimizer = ExtendedScipyOptimizerInterface(
            self.negative_data_loglikelihood, method="L-BFGS-B",
            var_list=gp_vars, var_to_bounds=var_to_bounds)
        # Optimize
        self.data_gp_optimizer.basinhopping(session, n_iter=50, stepsize=0.05)
        return

    def _build_states_bounds(self) -> None:
        """
        Builds the tensors for the normalized states that will containing the
        bounds for the constrained optimization.
        """
        # Tile the bounds to get the right dimensions
        state_lower_bounds = self.state_bounds[:, 0].reshape(self.n_states, 1)
        state_lower_bounds = np.tile(state_lower_bounds, [1, self.n_p])
        state_lower_bounds = (state_lower_bounds - self.system_data_means)\
            / self.system_data_std_dev
        state_lower_bounds = state_lower_bounds.reshape([self.n_states,
                                                         self.n_p])
        state_upper_bounds = self.state_bounds[:, 1].reshape(self.n_states, 1)
        state_upper_bounds = np.tile(state_upper_bounds, [1, self.n_p])
        state_upper_bounds = (state_upper_bounds - self.system_data_means)\
            / self.system_data_std_dev
        state_upper_bounds = state_upper_bounds.reshape([self.n_states,
                                                         self.n_p])
        self.state_lower_bounds = state_lower_bounds
        self.state_upper_bounds = state_upper_bounds
        return

    def _build_variables(self) -> None:
        """
        Builds the TensorFlow variables with the state values and the gamma
        that will later be optimized.
        """
        with tf.variable_scope('risk_main'):
            self.x = tf.Variable(self.system,
                                 dtype=tf.float64, trainable=True,
                                 name='states')
            if self.single_gp:
                self.log_gamma = tf.Variable(np.log(self.initial_gamma),
                                             dtype=tf.float64,
                                             trainable=self.train_gamma,
                                             name='log_gamma')
                self.gamma = tf.exp(self.log_gamma)\
                    * tf.ones([self.dimensionality, 1, 1], dtype=tf.float64)
            else:
                self.log_gamma =\
                    tf.Variable(np.log(self.initial_gamma)
                                * tf.ones([self.dimensionality, 1, 1],
                                          dtype=tf.float64),
                                trainable=self.train_gamma,
                                dtype=tf.float64,
                                name='log_gamma')
                self.gamma = tf.exp(self.log_gamma)
        return

    def _build_regularization_risk_term(self) -> tf.Tensor:
        """
        Build the first term of the risk, connected to regularization.
        :return: the TensorFlow Tensor that contains the term.
        """
        a_vector = tf.linalg.solve(
            self.gaussian_process.c_phi_matrices_noiseless,
            tf.expand_dims(self.x, -1))
        risk_term = 0.5 * tf.reduce_sum(self.x * tf.squeeze(a_vector))
        return tf.reduce_sum(risk_term)

    def _build_states_risk_term(self) -> tf.Tensor:
        """
        Build the second term of the risk, connected with the value of the
        states.
        :return: the TensorFlow Tensor that contains the term.
        """
        states_difference = self.system - self.x
        risk_term = tf.reduce_sum(states_difference * states_difference, 1)
        risk_term = risk_term * 0.5 / tf.squeeze(
            self.gaussian_process.likelihood_variances)
        return tf.reduce_sum(risk_term)

    def _build_derivatives_risk_term(self) -> tf.Tensor:
        """
        Build the third term of the risk, connected with the value of the
        derivatives.
        :return: the TensorFlow Tensor that contains the term.
        """
        # Compute model and data-based derivatives
        unnormalized_states = self.x * self.system_std_dev + self.system_means
        model_derivatives = tf.expand_dims(self.trainable.compute_gradients(
            unnormalized_states) / self.system_std_dev * self.t_std_dev, -1)
        data_derivatives =\
            self.gaussian_process.compute_posterior_derivative_mean(self.x)
        derivatives_difference = model_derivatives - data_derivatives
        # Compute log_variance on the derivatives
        post_variance =\
            self.gaussian_process.compute_posterior_derivative_variance() +\
            self.gamma * tf.expand_dims(tf.eye(self.n_points,
                                               dtype=tf.float64), 0)
        # Compute risk term
        a_vector = tf.linalg.solve(post_variance, derivatives_difference)
        risk_term = 0.5 * tf.reduce_sum(a_vector * derivatives_difference)
        return risk_term

    def _build_gamma_risk_term(self) -> tf.Tensor:
        """
        Build the term associated with gamma.
        :return: the TensorFlow Tensor that contains the terms
        """
        # Compute log_variance on the derivatives
        post_variance =\
            self.gaussian_process.compute_posterior_derivative_variance() +\
            self.gamma * tf.expand_dims(tf.eye(self.n_points,
                                               dtype=tf.float64), 0)
        risk_term = 0.5 * tf.linalg.logdet(post_variance)
        return tf.reduce_sum(risk_term)

    def _build_risk(self) -> None:
        """
        Build the risk tensor by summing up the single terms.
        """
        self.risk_term1 = self._build_regularization_risk_term()
        self.risk_term2 = self._build_states_risk_term()
        self.risk_term3 = self._build_derivatives_risk_term()
        self.risk = self.risk_term1 + self.risk_term2 + self.risk_term3
        if self.train_gamma:
            self.risk += self._build_gamma_risk_term()
        return

    def _build_optimizer(self) -> None:
        """
        Build the TensorFlow optimizer, wrapper to the scipy optimization
        algorithms.
        """
        # Extract the TF variables that get optimized in the risk minimization
        t_vars = tf.trainable_variables()
        risk_vars = [var for var in t_vars if 'risk_main' in var.name]
        # Dictionary containing the bounds on the TensorFlow Variables
        var_to_bounds = {risk_vars[0]: (self.trainable.parameter_lower_bounds,
                                        self.trainable.parameter_upper_bounds),
                         risk_vars[1]: (self.state_lower_bounds,
                                        self.state_upper_bounds)}
        if self.train_gamma:
            var_to_bounds[risk_vars[2]] = (self.gamma_bounds[0],
                                           self.gamma_bounds[1])
        self.risk_optimizer = ExtendedScipyOptimizerInterface(
            loss=self.risk, method=self.optimizer, var_list=risk_vars,
            var_to_bounds=var_to_bounds)
        return

    def build_model(self) -> None:
        """
        Builds Some common part of the computational graph for the optimization.
        """
        self.gaussian_process.build_supporting_covariance_matrices(
            self.t, self.t)
        self.negative_data_loglikelihood = \
            - self.gaussian_process.compute_average_log_likelihood(self.system)
        self._build_states_bounds()
        self._build_variables()
        self._build_risk()
        self._build_optimizer()
        return

    def _initialize_variables(self) -> None:
        """
        Initialize all the variables and placeholders in the graph.
        """
        self.init = tf.global_variables_initializer()
        return

    def _initialize_states_with_mean_gp(self, session: tf.Session) -> None:
        """
        Before optimizing the risk, we initialize the x to be the mean
        predicted by the Gaussian Process for an easier task later.
        :param session: TensorFlow session, used in the fit function.
        """
        mean_prediction = self.gaussian_process.compute_posterior_mean(
            self.system)
        assign_states_mean = tf.assign(self.x, tf.squeeze(mean_prediction))
        session.run(assign_states_mean)
        self.x = tf.clip_by_value(
            self.x, clip_value_min=tf.constant(self.state_lower_bounds),
            clip_value_max=tf.constant(self.state_upper_bounds))
        return

    def fit(self) -> [np.array, np.array, np.array]:
        """
        Fits the model.
        :return numpy arrays containing the system parameters theta, the gamma
        hyperparameters and the predicted states.
        """
        self._initialize_variables()
        session = tf.Session()
        with session:
            session.run(self.init)
            self._train_data_based_gp(session)
            self._initialize_states_with_mean_gp(session)
            if self.basinhopping:
                self.risk_optimizer.basinhopping(session,
                                                 **self.basinhopping_options)
            else:
                self.risk_optimizer.minimize(session)
            theta = session.run(self.trainable.theta)
            gamma = session.run(self.gamma).reshape(-1)
            x = session.run(tf.squeeze(self.x) * self.system_std_dev
                            + self.system_means)
        tf.reset_default_graph()
        return theta, gamma, x
