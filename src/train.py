import csv
import os
import numpy as np
import tensorflow as tf
import yaml
from tensorflow import keras


def main():
    p = yaml.safe_load(open("params.yaml"))["train"]
    tf.keras.utils.set_random_seed(p["seed"])

    tr = np.load("data/processed/train.npz")
    val = np.load("data/processed/val.npz")

    model = keras.Sequential([
        keras.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(p["dense_units"], activation="relu"),
        keras.layers.Dropout(p["dropout_rate"]),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    hist = model.fit(tr["x"], tr["y"],
                     validation_data=(val["x"], val["y"]),
                     epochs=p["epochs"], batch_size=p["batch_size"], verbose=2)

    os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")
    keys = list(hist.history.keys())
    with open("models/history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + keys)
        for i in range(len(hist.history[keys[0]])):
            w.writerow([i + 1] + [hist.history[k][i] for k in keys])


if __name__ == "__main__":
    main()