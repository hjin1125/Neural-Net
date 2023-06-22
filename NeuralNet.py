import numpy as np

def relu (x, derivative = False):
    if derivative: return np.where (x <= 0, 0, 1)
    else: return np.maximum(0, x)

def softmax(x):
    e_x = np.exp (x - np.max(x))
    return e_x / e_x.sum()

def cross_entropy_loss (predictions, targets):
    return -np.sum(targets * np.log(predictions + 1e-7)) / targets.shape[0]

def backward_propagation(x, targets, predictions, weights, biases):
    num_layers = len(weights)
    gradients = {}
    output_gradient = predictions - targets
    for layer in reversed(range(1, num_layers + 1)):
        gradient = output_gradient * relu (x[layer], derivative=True)
        gradients[f"dW{layer}"] = np.dot (x[layer-1].T, gradient)
        gradients[f"db{layer}"] = np.sum (gradient, axis=0)
        if layer > 1: # not input layer
            output_gradient = np.dot(gradient, weights[layer].T)
    return gradients