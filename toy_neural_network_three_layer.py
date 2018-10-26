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
Y = np.array([[0], [1], [1], [0]])
np.random.seed(1)

# initialize weights randomly
W1 = np.random.random((3, 4))
W2 = np.random.random((4, 1))

for iter in range(80000):
    z1 = np.dot(X,  W1)
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2)
    a2 = sigmoid(z2)

    de = a2 - Y
    if (iter % 10000) == 0:
        print("Error: {0}".format(np.mean(np.abs(de))))

    de_da2 = de * derivative_sigmoid(a2)
    de_dw2 = np.dot(a1.T, de_da2)

    de_da1 = np.dot(de_da2, W2.T) * derivative_sigmoid(a1)
    de_dw1 = np.dot(X.T, de_da1)

    W1 = W1 - de_dw1
    W2 = W2 - de_dw2

    # print("updated weights {0} {1}".format(W1, W2))

print("Final output after training:")
print(a2)
