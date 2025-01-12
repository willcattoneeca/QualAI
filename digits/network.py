import time
import random
import numpy as np

class FullyConnectedNN:
    def __init__(self, layer_sizes):
        # Initialize weights and biases using NumPy arrays
        self.weights = [np.random.uniform(-0.5, 0.5, (layer_sizes[i + 1], layer_sizes[i]))
                        for i in range(len(layer_sizes) - 1)]
        self.biases = [np.random.uniform(-0.5, 0.5, (size, 1)) for size in layer_sizes[1:]]
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def forward(self, x):
        # Store activations and z values for backpropagation
        self.activations = [x]
        self.z_values = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, self.activations[-1]) + b
            self.z_values.append(z)
            self.activations.append(self.sigmoid(z))
        return self.activations[-1]
    
    def backward(self, y):
        # Backpropagation
        deltas = [None] * len(self.weights)
        # Compute output layer delta
        delta = (self.activations[-1] - y) * self.sigmoid_derivative(self.z_values[-1])
        deltas[-1] = delta

        # Backpropagate through hidden layers
        for l in range(len(deltas) - 2, -1, -1):
            delta = np.dot(self.weights[l + 1].T, deltas[l + 1]) * self.sigmoid_derivative(self.z_values[l])
            deltas[l] = delta

        # Update weights and biases
        for l in range(len(self.weights)):
            # Update weights: gradient = deltas[l] @ activations[l].T
            self.weights[l] -= self.learning_rate * np.dot(deltas[l], self.activations[l].T) / y.shape[1]
            
            # Update biases: sum deltas over batch dimension
            self.biases[l] -= self.learning_rate * np.sum(deltas[l], axis=1, keepdims=True) / y.shape[1]
    
    def train(self, data, labels, epochs=10, batch_size=16, learning_rate=0.01):
        self.learning_rate = learning_rate
        num_batches = len(data) // batch_size
        print(f"Training started for {epochs} epochs, {num_batches} batches per epoch.")

        # Ensure data and labels are NumPy arrays
        data = np.array(data)
        labels = np.array(labels)
        
        for epoch in range(epochs):
            start_time = time.time()
            # Shuffle data
            indices = np.arange(len(data))
            np.random.shuffle(indices)
            data = data[indices]
            labels = labels[indices]
            epoch_loss = 0

            for batch_start in range(0, len(data), batch_size):
                x_batch = data[batch_start:batch_start + batch_size].T
                y_batch = labels[batch_start:batch_start + batch_size].T

                # Forward and backward pass
                self.forward(x_batch)
                self.backward(y_batch)

                # Compute batch loss
                batch_loss = -np.sum(y_batch * np.log(self.activations[-1] + 1e-15)) / y_batch.shape[1]
                epoch_loss += batch_loss

            epoch_loss /= num_batches
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{epochs} completed. Average Loss: {epoch_loss:.4f}. Time: {epoch_time:.2f}s")
    
    def evaluate(self, test_data, test_labels):
        correct = 0
        for x, y in zip(test_data, test_labels):
            prediction = self.forward(x.reshape(-1, 1))
            if np.argmax(prediction) == np.argmax(y):
                correct += 1
        print(f"Accuracy: {correct / len(test_data) * 100:.2f}%")

