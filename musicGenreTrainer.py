import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

def plot_history(history):

    fig, axs = plt.subplots(2)

    axs[0].plot(history.history["accuracy"], label="train accuracy", color="blue")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy", color="red")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")

    axs[1].plot(history.history["loss"], label="train error", color="blue")
    axs[1].plot(history.history["val_loss"], label="test error", color="red")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")

    plt.show()


def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    x = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return  x, y


DATA_PATH = "data_reduced.json"
x, y = load_data(DATA_PATH)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print(x.shape, y.shape)

call = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=5)
check = keras.callbacks.ModelCheckpoint("musicGenreRecogniser.keras", save_best_only=True, monitor='val_accuracy')

model = keras.Sequential([
    keras.layers.Input(shape=(x.shape[1], x.shape[2])),
    keras.layers.LSTM(256, return_sequences=True),
    keras.layers.LSTM(64),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer=keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=32, callbacks=[call, check])
plot_history(history)