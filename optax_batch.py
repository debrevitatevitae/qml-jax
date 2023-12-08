import jax
from jax import numpy as jnp
import jaxopt
import optax
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
    adam = optax.adam(learning_rate=eta)
    opt = jaxopt.OptaxSolver(opt=adam, fun=mse_loss)

    # define some initial parameters
    _, key = jax.random.split(key)
    init_params = jax.random.uniform(
        key, minval=0., maxval=2*jnp.pi, shape=(3,))

    # optimization loop with batching
    num_epochs = 100
    batch_size = 5

    def batch_data(key):
        _, key = jax.random.split(key)
        idxs_batch = jax.random.choice(key, num_samples, shape=(batch_size,))
        return key, X[idxs_batch], y[idxs_batch]

    num_batches_loss_eval = 5

    def compute_ave_mse_loss(key, params):
        ave_loss = 0.
        for _ in range(num_batches_loss_eval):
            key, X_batch, y_batch = batch_data(key)
            ave_loss += 1/num_batches_loss_eval * \
                mse_loss(params, X_batch, y_batch)
        return key, ave_loss

    opt_state = opt.init_state(init_params, X, y)
    params = init_params
    for ep in range(num_epochs):
        # Every some epochs report loss value averaged over some batches
        if ep % 10 == 0:
            key, loss_value = compute_ave_mse_loss(key, params)
            print(
                f"Epoch {ep}: MSE averaged over {num_batches_loss_eval} batches = {loss_value:.2e}")

        # select a batch
        key, X_batch, y_batch = batch_data(key)

        # update parameters and the optimizer state
        params, opt_state = opt.update(
            params, opt_state, X=X_batch, y=y_batch)

    # final parameters and loss value
    loss_value = mse_loss(params, X, y)
    print(f"Final MSE={loss_value:.2e}")
    print(f"Final params={params}")
