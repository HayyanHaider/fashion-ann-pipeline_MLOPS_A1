import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split


# ---- normalization ----
def normalize(x):
    x = x.astype("float32") / 255.0
    lo = x.min(axis=(1, 2), keepdims=True)
    hi = x.max(axis=(1, 2), keepdims=True)
    return (x - lo) / (hi - lo + 1e-7)    # per-image min-max


def main():
    """Normalize raw Fashion-MNIST arrays and split into train/val/test."""
    params = yaml.safe_load(open("params.yaml"))["preprocess"]
    d = np.load("data/raw/fashion_mnist_raw.npz")

    x_train, x_test = normalize(d["x_train"]), normalize(d["x_test"])
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, d["y_train"],
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=d["y_train"])

    os.makedirs("data/processed", exist_ok=True)
    np.savez_compressed("data/processed/train.npz", x=x_tr, y=y_tr)
    np.savez_compressed("data/processed/val.npz", x=x_val, y=y_val)
    np.savez_compressed("data/processed/test.npz", x=x_test, y=d["y_test"])
    print(f"train={x_tr.shape} val={x_val.shape} test={x_test.shape}")


if __name__ == "__main__":
    main()