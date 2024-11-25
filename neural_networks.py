import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

matplotlib.use("Agg")  # Use non-interactive backend

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def relu_derivative(x):
    return (x > 0).astype(float)


# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation="tanh"):
        np.random.seed(0)
        self.lr = lr
        self.activation_fn = activation
        self.weights_input_hidden = np.random.randn(input_dim, hidden_dim) * 0.1
        self.bias_hidden = np.zeros((1, hidden_dim))
        self.weights_hidden_output = np.random.randn(hidden_dim, output_dim) * 0.1
        self.bias_output = np.zeros((1, output_dim))
        self.hidden_activation = None
        self.output_activation = None

        if activation == "tanh":
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation == "relu":
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == "sigmoid":
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative

    def forward(self, X):
        self.input = X
        self.hidden_activation = self.activation(
            np.dot(X, self.weights_input_hidden) + self.bias_hidden
        )
        self.output_activation = sigmoid(
            np.dot(self.hidden_activation, self.weights_hidden_output)
            + self.bias_output
        )
        return self.output_activation

    def backward(self, X, y):
        # Calculate output layer error
        output_error = self.output_activation - y
        output_delta = output_error * sigmoid_derivative(self.output_activation)

        # Calculate hidden layer error
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.activation_derivative(self.hidden_activation)

        # Update weights and biases
        self.weights_hidden_output -= self.lr * np.dot(
            self.hidden_activation.T, output_delta
        )
        self.bias_output -= self.lr * np.sum(output_delta, axis=0, keepdims=True)
        self.weights_input_hidden -= self.lr * np.dot(X.T, hidden_delta)
        self.bias_hidden -= self.lr * np.sum(hidden_delta, axis=0, keepdims=True)


def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int).reshape(-1, 1)
    return X, y


# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y, steps_per_frame):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps
    for _ in range(steps_per_frame):
        mlp.forward(X)
        mlp.backward(X, y)

    # Plot hidden features
    hidden_features = mlp.hidden_activation
    ax_hidden.scatter(
        hidden_features[:, 0],
        hidden_features[:, 1],
        hidden_features[:, 2],
        c=y.ravel(),  # Ensure this is consistent with X
        cmap="bwr",
        alpha=0.7,
    )

    # Calculate and plot decision hyperplane
    x = np.linspace(-1, 1, 20)  # Hidden space x-coordinates
    y_grid = np.linspace(-1, 1, 20)  # Hidden space y-coordinates
    xx, yy = np.meshgrid(x, y_grid)

    # Solve for z based on the hyperplane equation
    weights = mlp.weights_hidden_output.ravel()  # Output weights
    bias = mlp.bias_output[0, 0]  # Output bias
    if weights[2] != 0:  # Avoid division by zero
        zz = (0.5 - bias - weights[0] * xx - weights[1] * yy) / weights[2]
        ax_hidden.plot_surface(
            xx, yy, zz, alpha=0.3, color="green", edgecolor="none"
        )  # Decision hyperplane

    ax_hidden.set_title(f"Hidden Space at Step {frame * steps_per_frame}")
    ax_hidden.set_xlim(-1, 1)
    ax_hidden.set_ylim(-1, 1)
    ax_hidden.set_zlim(-1, 1)

    # Plot decision boundary in input space
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = mlp.forward(grid).reshape(xx.shape)
    ax_input.contourf(xx, yy, predictions, alpha=0.7, cmap="bwr")
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="bwr", edgecolor="k")
    ax_input.set_title(f"Input Space at Step {frame * steps_per_frame}")
    ax_input.set_xlim(x_min, x_max)
    ax_input.set_ylim(y_min, y_max)

    # Visualize gradients
    ax_gradient.set_title(f"Gradients at Step {frame * steps_per_frame}")
    input_nodes = ["x1", "x2"]
    hidden_nodes = ["h1", "h2", "h3"]
    output_nodes = ["y"]

    # Plot input-to-hidden layer gradients
    for i, input_node in enumerate(input_nodes):
        for j, hidden_node in enumerate(hidden_nodes):
            weight_gradient = np.abs(
                mlp.weights_input_hidden[i, j]
            )  # Gradient magnitude
            ax_gradient.plot(
                [i, len(input_nodes) + j],
                [0, 1],
                linewidth=weight_gradient * 0.5,
                color="purple",
                alpha=0.7,
            )

    # Plot hidden-to-output layer gradients
    for j, hidden_node in enumerate(hidden_nodes):
        for k, output_node in enumerate(output_nodes):
            weight_gradient = np.abs(
                mlp.weights_hidden_output[j, k]
            )  # Gradient magnitude
            ax_gradient.plot(
                [len(input_nodes) + j, len(input_nodes) + len(hidden_nodes)],
                [1, 2],
                linewidth=weight_gradient * 0.5,
                color="purple",
                alpha=0.7,
            )

    # Add nodes for visualization with labels
    for i, node in enumerate(input_nodes):
        ax_gradient.scatter(i, 0, s=500, c="blue", zorder=5)
        ax_gradient.text(i, -0.2, node, ha="center", fontsize=10)
    for j, node in enumerate(hidden_nodes):
        ax_gradient.scatter(len(input_nodes) + j, 1, s=500, c="blue", zorder=5)
        ax_gradient.text(len(input_nodes) + j, 0.8, node, ha="center", fontsize=10)
    for k, node in enumerate(output_nodes):
        ax_gradient.scatter(
            len(input_nodes) + len(hidden_nodes) + k, 2, s=500, c="blue", zorder=5
        )
        ax_gradient.text(
            len(input_nodes) + len(hidden_nodes) + k,
            1.8,
            node,
            ha="center",
            fontsize=10,
        )

    ax_gradient.set_xlim(-1, len(input_nodes) + len(hidden_nodes) + len(output_nodes))
    ax_gradient.set_ylim(-0.5, 2.5)
    ax_gradient.axis("off")


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection="3d")
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    num_frames = step_num // 20  # Dynamically adjust frames based on step_num
    steps_per_frame = step_num // num_frames  # Steps to perform per frame

    ani = FuncAnimation(
        fig,
        partial(
            update,
            mlp=mlp,
            ax_input=ax_input,
            ax_hidden=ax_hidden,
            ax_gradient=ax_gradient,
            X=X,
            y=y,
            steps_per_frame=steps_per_frame,
        ),
        frames=num_frames,
        repeat=False,
    )

    # Save the animation
    ani.save(os.path.join(result_dir, "visualize.gif"), writer="pillow", fps=10)
    plt.close()


if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
