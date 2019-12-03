# ODIN
This repository contains code that implements the ODIN algorithm as presented in the paper:

> [**ODIN: ODE-Informed Regression for Parameter and State Inference in Time-Continuous Dynamical Systems.**](https://arxiv.org/abs/1902.06278)
> *Philippe Wenk\*, Gabriele Abbati\*, Michael A Osborne, Bernhard Schölkopf, Andreas Krause and Stefan Bauer*. Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI 20). New York, NY, USA. AAAI Press.
> \
> \*: equal contribution

## Code

The code provided is written in Python 3.6, and relies on the following libraries:
* [TensorFlow](https://www.tensorflow.org/)
* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/)

Some usage examples (the same ones used in the experimental section of the paper) can be found in the
[examples/](odin/examples/) directory. Comments in the code illustrate how it can be further applied to
generic ODE models not included here.
