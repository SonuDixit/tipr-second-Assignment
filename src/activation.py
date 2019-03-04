import numpy as np

def logsumexp(x):
    # found this way of numerical stability in scipy library
    x_max = np.max(x)
    return x_max + np.log(sum(np.exp(x - x_max)))
class Activation:
    def __init__(self):
        pass

    @staticmethod
    def swish(y):
        x = np.copy(y)
        return x * Activation.sigmoid(x)

    @staticmethod
    def grad_swish(y):
        x = np.copy(y)
        return Activation.sigmoid(x) + (x * Activation.grad_sigmoid(x))

    @staticmethod
    def softmax(y):
        x = np.copy(y)
        # print("softmax_input ",str(x))
        return np.exp(x-logsumexp(x))

    @staticmethod
    def sigmoid(y):
        x = np.copy(y)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def grad_sigmoid(y):
        x = np.copy(y)
        return Activation.sigmoid(x) * (1 - Activation.sigmoid(x))

    @staticmethod
    def relu(y):
        x = np.copy(y)
        zero_index = np.where(x < 0)
        # one_index = np.where(x>=0)
        x[zero_index] = 0.0
        return x

    @staticmethod
    def grad_relu(y):
        x = np.copy(y)
        one_index = np.where(x > 0.0)
        z = np.zeros(x.shape)
        z[one_index] = 1.0
        return z

    @staticmethod
    def tanh(y):
        x = np.copy(y)
        return 2 * Activation.sigmoid(2 * x) - 1

    @staticmethod
    def grad_tanh(y):
        x = np.copy(y)
        return 4 * Activation.grad_sigmoid(2 * x)

    @staticmethod
    def softplus(y):
        x = np.copy(y)
        return np.log(1 + np.exp(x))

    @staticmethod
    def grad_softplus(y):
        x = np.copy(y)
        return Activation.grad_sigmoid(x)