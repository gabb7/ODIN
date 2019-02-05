"""
Example script that runs the ODIN regression on the classic setting for the
protein transduction model.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Import libraries
import numpy as np
import tensorflow as tf
from odin import ProteinTransduction
from odin import TrainableProteinTransduction
from odin import ODIN


# Fix the random seeds for reproducibility
seed = 3298514
np.random.seed(seed)
tf.set_random_seed(seed)


# 1) Use the provided utilities class to simulate some noisy observations of
#    the protein transduction model

# We specify the time stamps for the observations
t_observations = np.array([0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0,
                          40.0, 50.0, 60.0, 80.0, 100.0])

protein_transduction_simulator = ProteinTransduction(
    true_param=[0.07, 0.6, 0.05, 0.3, 0.017, 0.3], noise_variance=1e-2**2)

system_obs, t_obs = protein_transduction_simulator.observe_at_t(
    initial_state=(1.0, 0.0, 1.0, 0.0, 0.0), initial_time=0.0,
    final_time=100.0, t_delta_integration=0.01, t_observations=t_observations)

n_states, n_points = system_obs.shape


# 2) Initialize the provided TrainableProteinTransduction class and set some
#    bounds for the theta variables

# Constraints on parameters
theta_bounds = np.array([[1e-8, 10.0], [1e-8, 10.0], [1e-8, 10.0], [1e-8, 10.0],
                         [1e-8, 10.0], [1e-8, 10.0]])

# Trainable object
trainable_protein_transduction = TrainableProteinTransduction(
    n_states, n_points, bounds=theta_bounds)


# 3) Run the actual ODIN regression by initializing the optimizer, building the
#    model and calling the fit() function

# Constraints on states
state_bounds = np.array([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0],
                         [0.0, 2.0]])

# ODIN optimizer
odin_optimizer = ODIN(trainable_protein_transduction,
                      system_obs,
                      t_obs,
                      gp_kernel='Sigmoid',  # For PT we use the Sigmoid kernel
                      optimizer='L-BFGS-B',  # L-BFGS-B optimizer for the bounds
                      initial_gamma=1e-1,  # initial gamma value
                      train_gamma=True,  # gamma will be trained as well
                      gamma_bounds=(1e-6, 10.0),  # bounds on gamma
                      state_bounds=state_bounds,  # Pass the state bounds
                      single_gp=False,  # Here we use one GP per state
                      basinhopping=True,  # Here we do use basinhopping
                      time_normalization=False,  # Better fit if off (empirical)
                      state_normalization=True)  # states normalization on

# Build the model
odin_optimizer.build_model()

# Fit the model
final_theta, final_gamma, final_x = odin_optimizer.fit()
