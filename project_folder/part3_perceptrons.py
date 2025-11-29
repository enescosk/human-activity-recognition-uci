# Part 3) Perceptrons (OR, NAND, AND -> XOR):
import numpy as np

# 1. Defining a simple perceptron class with bias, learning rate, and training loop:
class Perceptron:
    def __init__(self, inputSize, lr=0.1, epochs=20):
        self.lr = lr
        self.epochs = epochs
        self.weights = np.zeros(inputSize + 1)  # bias = 1
    
    def activation(self, x):
        return np.where(x >= 0, 1, 0)
    
    def predict(self, x):
        z = np.dot(x, self.weights[1:]) + self.weights[0]
        return self.activation(z)
    
    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                y_pred = self.predict(X[i])
                update = self.lr * (y[i] - y_pred)
                self.weights[1:] += update * X[i]
                self.weights[0] += update

# 2. Defining 2-bit binary input samples:
X = np.array([[0,0],[0,1],[1,0],[1,1]])

# 3. Training the OR perceptron:
labelOR = np.array([0,1,1,1])
modelOR = Perceptron(2)
modelOR.train(X, labelOR)

# 4. Training the AND perceptron:
labelAND = np.array([0,0,0,1])
modelAND = Perceptron(2)
modelAND.train(X, labelAND)

# 5. Training the NAND perceptron:
labelNAND = np.array([1,1,1,0])
modelNAND = Perceptron(2)
modelNAND.train(X, labelNAND)

# 6. Creating XOR using (A NAND B) AND (A OR B):
outputXOR = []
for x in X:
    outputNAND = modelNAND.predict(x)
    outputOR   = modelOR.predict(x)
    inputXOR = np.array([outputNAND, outputOR])
    predictedXOR = modelAND.predict(inputXOR)
    outputXOR.append(predictedXOR)

# 7. Printing OR, NAND, AND, and XOR outputs for each input combination:
print("\nInputs | OR  NAND  AND | XOR")
print("------------------------------")
for i in range(len(X)):
    print(f"{X[i]} |  {modelOR.predict(X[i])}    {modelNAND.predict(X[i])}     {modelAND.predict(X[i])}   |  {outputXOR[i]}")

# 8. Computing XOR accuracy using expected labels:
expectedXOR = np.array([0,1,1,0])
acc = np.mean(np.array(outputXOR) == expectedXOR)
print(f"\nAccuracy for XOR using 3 perceptrons: {acc*100:.1f}%")
