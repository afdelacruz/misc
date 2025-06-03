import argparse
import joblib
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from PIL import Image


def train(model_path: str):
    """Train logistic regression to classify shoes vs non-shoes."""
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)
    y = y.astype(int)
    shoe_labels = [7, 9]  # sneaker and ankle boot classes in Fashion-MNIST
    y_binary = np.isin(y, shoe_labels).astype(int)

    clf = LogisticRegression(max_iter=100, n_jobs=-1)
    clf.fit(X, y_binary)
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")


def predict(image_path: str, model_path: str):
    """Predict whether the image is a shoe using a trained model."""
    clf = joblib.load(model_path)
    img = Image.open(image_path).convert('L').resize((28, 28))
    X = np.array(img).reshape(1, -1)
    pred = clf.predict(X)[0]
    label = 'shoe' if pred == 1 else 'not shoe'
    print(label)
    return label


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run the shoe classifier")
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--model', default='shoe_model.pkl', help='Output model path')

    pred_parser = subparsers.add_parser('predict', help='Predict on an image')
    pred_parser.add_argument('image', help='Path to image file')
    pred_parser.add_argument('--model', default='shoe_model.pkl', help='Path to trained model')

    args = parser.parse_args()
    if args.command == 'train':
        train(args.model)
    elif args.command == 'predict':
        if not Path(args.model).exists():
            raise FileNotFoundError(f"Model file {args.model} not found. Train it first using 'train'.")
        predict(args.image, args.model)


if __name__ == '__main__':
    main()
