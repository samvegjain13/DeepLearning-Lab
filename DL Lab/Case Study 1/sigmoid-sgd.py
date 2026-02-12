import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 784) / 255.0
x_test  = x_test.reshape(-1, 784) / 255.0

model = Sequential([
    Dense(128, activation='sigmoid', input_shape=(784,)),
    Dense(10, activation='softmax')
])

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
print("\nTest Accuracy:", test_acc)

# 1️⃣ Accuracy
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Train', 'Validation'])
plt.title("Sigmoid - Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig("Case Study 1/plots/sigmoid-sgd-accuracy.png")
plt.close()

# 2️⃣ Loss (Train vs Val)
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train', 'Validation'])
plt.title("Sigmoid - Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("Case Study 1/plots/sigmoid-sgd-loss.png")
plt.close()

# 3️⃣ Loss vs Epochs
plt.figure()
plt.plot(history.history['loss'])
plt.title("Sigmoid - Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("Case Study 1/plots/sigmoid-sgd-loss-epoch.png")
plt.close()
