import jax
from jax import numpy as jnp
import jaxopt
import pennylane as qml

jax.config.update("jax_enable_x64", True)


if __name__ == '__main__':
    key = jax.random.PRNGKey(seed=42)

    # create some sine data at random as jax arrays
    num_samples = 20
    noise_amplitude = 0.05
    _, key = jax.random.split(key)
    X = jax.random.uniform(key, minval=0., maxval=jnp.pi, shape=(num_samples,))
    noise = noise_amplitude * jnp.ones_like(X)
    y = jnp.sin(X) + noise

    # define a VQC with PennyLane, jax and jax.jit
    num_qubits = 1
    dev = qml.device("default.qubit.jax", wires=num_qubits)
    # dev = qml.device("default.qubit.jax", interface='jax', wires=num_qubits)

    @jax.jit
    @qml.qnode(dev, interface='jax')
    def vqc(x, params):
        qml.RY(x, wires=0)
        qml.Rot(params[0], params[1], params[2], wires=0)
        return qml.expval(qml.PauliZ(0))

    # define a MSE loss with args = (params, X, y)
    def mse_loss(params, X, y):
        errs_squared = jnp.array(
            [(vqc(x_i, params) - y_i) ** 2 for x_i, y_i in zip(X, y)])
        return 1/len(y) * jnp.sum(errs_squared)

    # define a JaxOpt gradient descent optmizer
    eta = 0.1
    opt = jaxopt.GradientDescent(mse_loss, stepsize=eta, maxiter=100)

    # define some initial parameters
    _, key = jax.random.split(key)
    init_params = jax.random.uniform(
        key, minval=0., maxval=2*jnp.pi, shape=(3,))

    # optimize with optimizer.run syntax
    res = opt.run(init_params, X=X, y=y)

    # print the final state and parameters
    print(res.params)
    print(res.state)
