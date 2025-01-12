import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.regularizers import l2
import numpy as np

class FullyConnectedNN:
    def __init__(self, layer_sizes):
        """
        Initializes the FullyConnectedNN class and builds the TensorFlow model.

        Args:
            layer_sizes: List of integers representing the number of neurons in each layer.
        """
        self.input_size = layer_sizes[0]
        self.output_size = layer_sizes[-1]

        # Build the model
        self.model = Sequential()
        self.model.add(tf.keras.Input(shape=(self.input_size,)))  # Input layer

        # Hidden layers with L1 and L2 regularization
        for size in layer_sizes[1:-1]:
            self.model.add(Dense(size, activation='relu', kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5)))
            self.model.add(Dropout(0.1))

        # Output layer
        self.model.add(Dense(self.output_size, activation='softmax'))

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, train_X, train_y, epochs=50, batch_size=64):
        """
        Trains the neural network on the provided training data.

        Args:
            train_X: Training features (numpy array).
            train_y: Training labels (one-hot encoded numpy array).
            epochs: Number of epochs to train for.
            batch_size: Batch size to use during training.
        """
        self.model.fit(train_X, np.array(train_y), epochs=epochs, batch_size=batch_size, validation_split=0.1)

    def evaluate(self, test_X, test_y):
        """
        Evaluates the model on the test set.

        Args:
            test_X: Test features (numpy array).
            test_y: Test labels (one-hot encoded numpy array).
        """
        loss, accuracy = self.model.evaluate(test_X, np.array(test_y), verbose=2)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def forward(self, input_data):
        """
        Performs a forward pass and predicts the probabilities for each class.

        Args:
            input_data: Input data (numpy array).

        Returns:
            Predicted probabilities for each class.
        """
        return self.model.predict(input_data.T)  # Transpose to match (batch_size, features) format



class ConvNet:
    def __init__(self):
        """
        Initializes the ConvNet class and builds the TensorFlow convolutional model.
        """
        # Build the model
        self.model = Sequential()

        # Convolutional layers
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=l2(1e-4)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(1e-4)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.2))

        # Flattening the output of the convolutional layers
        self.model.add(Flatten())

        # Fully connected layers
        self.model.add(Dense(128, activation='relu', kernel_regularizer=l2(1e-4)))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(10, activation='softmax'))

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, train_X, train_y, epochs=50, batch_size=64, validation_data=None):
        """
        Trains the ConvNet on the provided training data.

        Args:
            train_X: Training features (numpy array of shape (N, 28, 28, 1)).
            train_y: Training labels (one-hot encoded numpy array).
            epochs: Number of epochs to train for.
            batch_size: Batch size to use during training.
            validation_data: Tuple (val_X, val_y) for validation.
        """
        self.model.fit(train_X, np.array(train_y), epochs=epochs, batch_size=batch_size, validation_split=0.1)

    def evaluate(self, test_X, test_y):
        """
        Evaluates the model on the test set.

        Args:
            test_X: Test features (numpy array of shape (N, 28, 28, 1)).
            test_y: Test labels (one-hot encoded numpy array).
        """
        loss, accuracy = self.model.evaluate(test_X, np.array(test_y), verbose=2)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def forward(self, input_data):
        """
        Performs a forward pass and predicts the probabilities for each class.

        Args:
            input_data: Input data (numpy array of shape (N, 28, 28, 1)).

        Returns:
            Predicted probabilities for each class.
        """
        return self.model.predict(input_data)
