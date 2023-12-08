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
    key = jax.random.PRNGKey(seed=42)

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

    # convert the data into jax.numpy array
    X_train_scaled = jnp.array(X_train_scaled)
    y_train = jnp.array(y_train)
    X_test_scaled = jnp.array(X_test_scaled)
    y_test = jnp.array(y_test)

    # define a VQC with PennyLane, jax and jax.jit
    dev = qml.device("default.qubit.jax", wires=num_features)

    @jax.jit
    @qml.qnode(dev, interface='jax')
    def vqc(x, params):
        for i in range(num_features):
            qml.RY(x[i], wires=i)
            qml.Rot(params[i, 0], params[i, 1], params[i, 2], wires=i)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))

    def clf_model(x, params):
        # include [0, 1] rescaling of the prediction
        return 0.5 * (1 + vqc(x, params))

    # define a log-loss with args = (params, X, y)
    def cross_entropy_loss(params, X, y):
        eps = 1e-15  # small constant to avoid log(0)
        y_pred = jnp.array([clf_model(x_i, params) for x_i in X])
        # clip the predictions to avoid log(0)
        y_pred = jnp.clip(y_pred, eps, 1 - eps)
        loss = - (y * jnp.log(y_pred) + (1 - y) * jnp.log(1 - y_pred))
        return jnp.mean(loss)

    # define a JaxOpt gradient descent optmizer
    eta = 0.1
    adam = optax.adam(learning_rate=eta)
    opt = jaxopt.OptaxSolver(opt=adam, fun=cross_entropy_loss)

    # define some initial parameters
    _, key = jax.random.split(key)
    init_params = jax.random.uniform(
        key, minval=0., maxval=2*jnp.pi, shape=(num_features, 3))

    # optimization loop with batching
    num_epochs = 1000
    batch_size = 20

    def batch_data(key):
        _, key = jax.random.split(key)
        idxs_batch = jax.random.choice(key, num_samples, shape=(batch_size,))
        return key, X_train_scaled[idxs_batch], y_train[idxs_batch]

    num_batches_loss_eval = 5

    def compute_ave_cross_entropy_loss(key, params):
        ave_loss = 0.
        for _ in range(num_batches_loss_eval):
            key, X_batch, y_batch = batch_data(key)
            ave_loss += 1/num_batches_loss_eval * \
                cross_entropy_loss(params, X_batch, y_batch)
        return key, ave_loss

    opt_state = opt.init_state(init_params, X, y)
    params = init_params
    for ep in range(num_epochs):
        # Every some epochs report loss value averaged over some batches
        if ep % 10 == 0:
            key, loss_value = compute_ave_cross_entropy_loss(key, params)
            print(
                f"Epoch {ep}: log-loss averaged over {num_batches_loss_eval} batches = {loss_value:.2e}")

        # select a batch
        key, X_batch, y_batch = batch_data(key)

        # update parameters and the optimizer state
        params, opt_state = opt.update(
            params, opt_state, X=X_batch, y=y_batch)

    # final parameters and loss value
    loss_value = cross_entropy_loss(params, X_train_scaled, y_train)
    print(f"Final log-loss={loss_value:.2e}")
    print(f"Final params={params}")

    # comparison over test samples
    preds_test = jnp.array([clf_model(x_i, params) for x_i in X_test_scaled])
    preds_test_labels = (preds_test > 0.5).astype(jnp.int16)

    accuracy_test = accuracy_score(y_test, preds_test_labels)
    print(f"Accuracy over test dataset = {accuracy_test:.3f}")
