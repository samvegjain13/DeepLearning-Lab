import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape 28x28 images into 784 vector and normalize (0-1 scaling)
x_train = x_train.reshape(-1, 784) / 255.0
x_test  = x_test.reshape(-1, 784) / 255.0

print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),  # Hidden Layer
    Dense(64, activation='relu'),                       # Additional Hidden Layer
    Dense(10, activation='softmax')                     # Output Layer (10 classes)
])

model.summary()

model.compile(
    optimizer=SGD(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

test_loss, test_acc = model.evaluate(x_test, y_test)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_acc)

# Accuracy Plot
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['Train', 'Validation'])
plt.savefig("Case Study 1/plots/relu-sgd-accuracy.png")
plt.show()

# Loss Plot (Train vs Validation)
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['Train', 'Validation'])
plt.savefig("Case Study 1/plots/relu-sgd-loss.png")
plt.show()

# Loss vs Epochs (Separate)
plt.figure()
plt.plot(history.history['loss'])
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("Case Study 1/plots/relu-sgd-loss-epoch.png")
plt.show()