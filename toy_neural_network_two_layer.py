import numpy as np


# sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# derivative of sigmoid function
def derivative_sigmoid(s):
    return s * (1-s)

# Input dataset
X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

# Output dataset
Y = np.array([[0], [0], [1], [1]])

print(Y)
np.random.seed(1)

# initialize weights randomly
W = np.random.random((3, 1))


for iter in range(10000):
    z = np.dot(X, W)
    a = sigmoid(z)

    de = a - Y
    sigmoid_prime = de * derivative_sigmoid(a)

    d = np.dot(sigmoid_prime.T, X).T
    W = W - d


print("Final output after training:")
print(a)
