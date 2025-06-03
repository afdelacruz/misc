# Shoe Classifier

This repository contains a simple pipeline for training and running a lightweight shoe classifier.

The classifier is based on scikit-learn's `LogisticRegression` and is trained using the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset, which provides 28x28 grayscale images for various clothing items including two categories of shoes.

## Training

```
python3 shoe_classifier.py train --model shoe_model.pkl
```

The script downloads Fashion-MNIST using `fetch_openml`, converts the labels to a binary **shoe** vs **not shoe** target and trains a logistic regression model. The trained model is saved as `shoe_model.pkl` by default.

## Prediction

```
python3 shoe_classifier.py predict path/to/image.png --model shoe_model.pkl
```

Given an input image, the script resizes it to 28x28 grayscale pixels and outputs `shoe` or `not shoe`.
