import jax
from jax import numpy as jnp
import jaxopt
import numpy as np
import optax
import pennylane as qml
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

jax.config.update("jax_enable_x64", True)


if __name__ == '__main__':
    key = jax.random.PRNGKey(seed=40)

    # load the iris dataset and select instances from the first 2 classes
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    setosa_versicolor = np.where(y < 2)
    X = X[setosa_versicolor]
    y = y[setosa_versicolor]
    num_samples = X.shape[0]
    num_features = X.shape[1]

    # split the dataset and rescale in [0, \pi)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # convert the data into jax.numpy array and rescale labels to {-1, +1}
    X_train_scaled = jnp.array(X_train_scaled)
    y_train = jnp.array([2*y_i - 1 for y_i in y_train])
    X_test_scaled = jnp.array(X_test_scaled)
    y_test = jnp.array([2*y_i - 1 for y_i in y_test])

    # define layer and embedding
    def layer(x, params, wires, i0=0, inc=1):
        """Taken from https://github.com/thubregtsen/qhack"""
        i = i0
        for j, wire in enumerate(wires):
            qml.Hadamard(wires=[wire])
            qml.RZ(x[i % len(x)], wires=[wire])
            i += inc
            qml.RY(params[0, j], wires=[wire])

        qml.broadcast(unitary=qml.CRZ, pattern="ring",
                      wires=wires, parameters=params[1])

    def embedding(x, params, wires):
        """Adapted from https://github.com/thubregtsen/qhack"""
        inc = 1
        for j, layer_params in enumerate(params):
            layer(x, layer_params, wires, i0=j * len(wires))
        # encode data one last time to avoid cancellations in the kernel circuit
        i = len(params) * len(wires)
        for wire in wires:
            qml.Hadamard(wires=[wire])
            qml.RZ(x[i % len(x)], wires=[wire])
            i += inc

    # define the quantum kernel with PennyLane, jax and jax.jit
    dev = qml.device("default.qubit.jax", wires=num_features)
    # use as many qubits as there are features
    wires = list(range(num_features))

    @jax.jit
    @qml.qnode(dev, interface='jax')
    def kernel(x1, x2, params):
        """x1, x2 and params must be JAX arrays"""
        embedding(x1, params, wires)
        qml.adjoint(embedding)(x2, params, wires)
        return qml.expval(qml.Projector([0]*len(wires), wires=wires))

    # kta-based parametrised loss
    def kta_loss(params, X, y):
        # compute the kernel matrix
        N = len(y)
        K = jnp.eye(N)
        for i in range(N-1):
            for j in range(i+1, N):
                # Compute kernel value and fill in both (i, j) and (j, i) entries
                K_i_j = kernel(X[i], X[j], params)
                K = K.at[i, j].set(K_i_j)
                K = K.at[j, i].set(K_i_j)

        # compute the target kernel
        T = jnp.outer(y, y)

        # compute polarity
        polarity = jnp.sum(K * T)

        # normalise
        kta = polarity / (N * jnp.sum(K * K))

        return -kta

    # define a JaxOpt gradient descent optmizer
    eta = 0.1
    adam = optax.adam(learning_rate=eta)
    opt = jaxopt.OptaxSolver(opt=adam, fun=kta_loss)

    # define some initial parameters
    num_layers = 2
    _, key = jax.random.split(key)
    init_params = jax.random.uniform(
        key, minval=0., maxval=2*jnp.pi, shape=(num_layers, 2, num_features))

    # optimization loop with batching
    num_epochs = 50
    batch_size = 5

    def batch_data(key):
        _, key = jax.random.split(key)
        idxs_batch = jax.random.choice(key, num_samples, shape=(batch_size,))
        return key, X_train_scaled[idxs_batch], y_train[idxs_batch]

    num_batches_loss_eval = 5

    def compute_ave_kta_loss(key, params):
        ave_loss = 0.
        for _ in range(num_batches_loss_eval):
            key, X_batch, y_batch = batch_data(key)
            ave_loss += 1/num_batches_loss_eval * \
                kta_loss(params, X_batch, y_batch)
        return key, ave_loss

    opt_state = opt.init_state(init_params, X, y)
    params = init_params
    for ep in range(num_epochs):
        # Every some epochs report loss value averaged over some batches
        if ep % 10 == 0:
            key, loss_value = compute_ave_kta_loss(key, params)
            print(
                f"Epoch {ep}: kta averaged over {num_batches_loss_eval} batches = {-loss_value:.4f}")

        # select a batch
        key, X_batch, y_batch = batch_data(key)

        # update parameters and the optimizer state
        params, opt_state = opt.update(
            params, opt_state, X=X_batch, y=y_batch)

    # final parameters and loss value
    loss_value = kta_loss(params, X_train_scaled, y_train)
    print(f"Final KTA={-loss_value:.4f}")
    # print(f"Final params={params}")
