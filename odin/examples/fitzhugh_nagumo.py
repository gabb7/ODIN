"""
Example script that runs the ODIN regression on the FitzHugh - Nagumo neuron
model.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""


# Import libraries
import numpy as np
import tensorflow as tf
from odin import FitzHughNagumo
from odin import TrainableFitzHughNagumo
from odin import ODIN


# Fix the random seeds for reproducibility
seed = 3298514
np.random.seed(seed)
tf.set_random_seed(seed)


# 1) Use the provided utilities class to simulate some noisy observations of
#    the FitzHugh-Nagumo model

# Simulate the data
fitzhugh_nagumo_simulator = FitzHughNagumo(true_param=(0.2, 0.2, 3.0),
                                           noise_variance=0.0,
                                           stn_ratio=100.0)
system_obs, t_obs =\
    fitzhugh_nagumo_simulator.observe(initial_state=(-1.0, 1.0),
                                      initial_time=0.0,
                                      final_time=10.0,
                                      t_delta_integration=0.01,
                                      t_delta_observation=0.5)
n_states, n_points = system_obs.shape


# 2) Initialize the provided TrainableFitzHughNagumo class and set some bounds
#    for the theta variables

# Constraints on parameters
theta_bounds = np.array([[0.0, 100.0], [0.0, 100.0], [0.0, 100.0]])

trainable_fitzhugh_nagumo = TrainableFitzHughNagumo(n_states,
                                                    n_points,
                                                    bounds=theta_bounds)

# 3) Run the actual ODIN regression by initializing the optimizer, building the
#    model and calling the fit() function

# ODIN optimizer
odin_optimizer = ODIN(trainable_fitzhugh_nagumo,
                      system_obs,
                      t_obs,
                      gp_kernel='Matern52',  # For FHN we use the Matern kernel
                      optimizer='L-BFGS-B',  # L-BFGS-B optimizer for the bounds
                      initial_gamma=1.0,  # initial gamma value
                      train_gamma=True,  # gamma will be trained as well
                      single_gp=False,  # Here we use one GP per state
                      basinhopping=True,  # Basinhopping activated
                      basinhopping_options={'n_iter': 10},  # Set 10 iterations
                      time_normalization=True,  # time normalization on
                      state_normalization=True)  # states normalization on

# Build the model
odin_optimizer.build_model()

# Fit the model
final_theta, final_gamma, final_x = odin_optimizer.fit()
