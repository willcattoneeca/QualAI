import os
from network_tensorflow import FullyConnectedNN, ConvNet
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def predict_and_visualize(nn, digit_image, true_label):
    """
    Predicts the label of a digit using the neural network and visualizes it.
    
    Args:
        nn: The trained FullyConnectedNN instance.
        digit_image: The input image (28x28 array) of the digit.
        true_label: The true label of the digit.
    """
    digit_flat = digit_image.reshape(-1, 1) / 255.0
    prediction = nn.forward(digit_flat)
    predicted_label = np.argmax(prediction)
    plt.imshow(digit_image, cmap='gray')
    plt.title(f"True Label: {true_label}, Predicted Label: {predicted_label}")
    plt.axis('off')
    plt.show()

def show(i):
    digit_image = test_X[i]        # Original 28x28 image
    true_label = test_y[i]         # True label
    predict_and_visualize(nn, digit_image, true_label)

(train_X, train_y), (test_X, test_y) = mnist.load_data()
# Normalize pixel values to range [0, 1]
train_X = train_X / 255.0
test_X = test_X / 255.0

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
#plt.show()

# Flatten the images
train_X_flat = train_X.reshape((train_X.shape[0], 28 * 28))
test_X_flat = test_X.reshape((test_X.shape[0], 28 * 28))

# One-hot encode labels
def one_hot_encode(labels, num_classes=10):
    return [[1 if i == label else 0 for i in range(num_classes)] for label in labels]

train_y_one_hot = one_hot_encode(train_y)
test_y_one_hot = one_hot_encode(test_y)

print("Preprocessing complete.")
print("Flattened training data shape:", train_X_flat.shape)
print("Flattened test data shape:", test_X_flat.shape)


# Define the neural network structure
layer_sizes = [784, 128, 64, 10]  # Input -> Hidden Layer 1 -> Hidden Layer 2 -> Output
#layer_sizes = [784, 256, 128, 64, 10] # Try one more hidden layer, slightly more features
if False:
    nn = FullyConnectedNN(layer_sizes)
    # Train the network
    #nn.train(train_X_flat, train_y_one_hot, epochs=50, batch_size=32, learning_rate=0.1)
    nn.train(train_X_flat, train_y_one_hot, epochs=50, batch_size=32)
    # Evaluate on the test set
    nn.evaluate(test_X_flat, test_y_one_hot)
else:
    nn = ConvNet()
    nn.train(train_X, train_y_one_hot, epochs=50, batch_size=32)
    # Evaluate on the test set
    nn.evaluate(test_X, test_y_one_hot)


# Example: Visualize predictions for 5 random test digits
for i in [0, 42, 100, 999, 1234]:
    show(i)
