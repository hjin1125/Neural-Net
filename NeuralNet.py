import numpy as np

def relu (x): return np.maximum(0, x)

def softmax (x): 
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def neural_network (x, weights):
    hidden_layer1 = relu (np.dot (x, weights['W1']) + weights['b1'])
    hidden_layer2 = relu (np.dot (hidden_layer1, weights['W2']) + weights['b2'])
    output_layer = softmax (np.dot (hidden_layer2, weights['W3']) + weights['b3'])
    return output_layer

num_points = 100
test_data = np.random.rand (num_points, 2)

input_num = 2 # x, y
hidden1_num = 6 # 6 neurons
hidden2_num = 4 # 4 neurons
output_num = 2 # far/close

weights = {
    'W1': np.random.randn (input_num, hidden1_num),
    'b1': np.zeros (hidden1_num),
    'W2': np.random.randn (hidden1_num, hidden2_num),
    'b2': np.zeros (hidden2_num),
    'W3': np.random.randn (hidden2_num, output_num),
    'b3': np.zeros (output_num)
}

predictions = neural_network (test_data, weights)

for i in range(num_points):
    point = test_data[i]
    prediction = predictions[i]
    if prediction[0] > prediction[1]: category = "far from middle"
    else: category = "close to middle"
    print(f"{point}: {category}")