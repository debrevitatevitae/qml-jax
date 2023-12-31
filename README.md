# qml-jax
Tests with PennyLane and JAX.

## Description of files
- `jaxopt_run.py`: simple sine-function learning of a VQC with `pennylane`, `jax.jit` and `jaxopt.GradientDescent`. The `.run()` method is called on the optimizer.
- `jaxopt_optimization_loop.py`: same as `jaxopt.py`, but breaks down the optmization loop, rather than calling the `run()` method on the optimizer object.
- `optax_full_set.py`: same as `jaxopt_optimization_loop.py`, but uses `optax.adam` as optimizer and updates with the full dataset.
- `optax_batch.py`: same as `optax_full_set.py` but stochastic: uses a data iterator to batch `X` and `y`.
- `optax_classification.py`: classification of the iris dataset with `optax`. Optimization part is the same as `optax_batch.py`, but the data now is preprocessed and converted from NumPy to JAX.
- `optax_kta.py`: optimizes the kernel alignment of a quantum kernel wrt the (binary) iris dataset.
