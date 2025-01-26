Here is a professional README file for your GitHub repository based on the provided script:

---

# Image Classification Using CNN

This project demonstrates how to build, train, and evaluate a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. It also includes functionality for predicting the class of custom images using the trained model.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Model Deployment](#model-deployment)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction
Convolutional Neural Networks (CNNs) are a class of deep learning models specifically designed for image data. This project uses the CIFAR-10 dataset to classify images into 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

---

## Dataset
The [CIFAR-10 dataset] is a collection of 60,000 32x32 color images divided into 10 classes, with 6,000 images per class. It is commonly used for image classification tasks.

---

## Features
- **Model Architecture**: A sequential CNN model with three convolutional layers, max-pooling layers, and dense layers.
- **Training & Evaluation**: Trains the CNN on a reduced dataset for 10 epochs and evaluates its accuracy.
- **Image Prediction**: Includes a function to predict the class of custom images.
- **Model Saving**: Saves the trained model in `.h5` format for later use.

---

## Prerequisites
Ensure the following packages are installed:
- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- OpenCV

Install the dependencies using:
```bash
pip install tensorflow numpy matplotlib opencv-python
```


---

## Usage

### 1. Training the CNN
The script trains a CNN on the CIFAR-10 dataset. By default, the dataset is normalized, and a reduced subset of 30,000 training images and 4,000 testing images is used.

To run the training:
```bash
python image_classification_using_cnn.py
```

### 2. Predicting Custom Images
Update the `image_path` list with paths to your custom images. The `predict_image()` function resizes the input image to 32x32 and predicts its class.

### 3. Saving the Model
The trained model is saved as `cnn-image-classifier.h5` in the specified directory.


---

## Model Deployment
The trained model can be loaded using TensorFlow for further usage:
```python
from tensorflow.keras.models import load_model

model = load_model('path_to_model/cnn-image-classifier.h5')
```

---

## Contributing
Contributions are welcome! Please fork this repository and submit a pull request with your improvements or new features.

---
