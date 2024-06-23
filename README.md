# Dogs vs. Cats Classification

## Overview
This project involves creating an algorithm to classify images as containing either a dog or a cat. The primary goal is to train a model on a given dataset and predict labels for a separate test dataset.

## Dataset Description
- The training dataset contains 25,000 images, split equally between cats and dogs (12,500 images each).
- The task is to train an algorithm on these images and predict labels for the test dataset, with labels being 1 for dogs and 0 for cats.

## Project Steps

### 1. Import Libraries
The following libraries are used in this project:
- **Pandas**: For loading and handling data.
- **Numpy**: For numerical computations.
- **Matplotlib & Seaborn**: For data visualization.
- **OpenCV & PIL**: For image processing.
- **Scikit-learn**: For data preprocessing and model evaluation.
- **TensorFlow & Keras**: For building and training the CNN model.

### 2. Data Extraction
- The dataset is provided in zip files containing images of cats and dogs.
- Python's `zipfile` module is used to extract the contents into a specified directory.

### 3. Data Exploration
- Visual exploration of images is conducted to understand the dataset better.
- Example images of both cats and dogs are displayed.

### 4. Data Preparation
- The dataset is split into training, validation, and test sets using an 80-10-10 split.
- Class distribution is checked and visualized to ensure a balanced dataset.

### 5. Directory Structure
- Directories for training and test sets are created, with subdirectories for cats and dogs.

### 6. Image Data Generator
- Image augmentation techniques are applied using TensorFlow's `ImageDataGenerator` to improve model generalization.
- Different generators are created for training, validation, and test datasets.

### 7. Model Building
- A Convolutional Neural Network (CNN) is built using Keras' Sequential API.
- The model consists of multiple convolutional, pooling, and dropout layers, followed by fully connected layers.
- The final layer uses softmax activation for binary classification.

### 8. Model Compilation and Training
- The model is compiled with appropriate loss functions, optimizers, and metrics.
- Callbacks such as `ReduceLROnPlateau` and `EarlyStopping` are used to optimize training.

### 9. Model Evaluation
- The model's performance is evaluated on the validation and test datasets.
- Confusion matrices and classification reports are generated to assess accuracy and other metrics.

## Conclusion
This project demonstrates the process of building a robust image classification model to distinguish between cats and dogs. It involves data preprocessing, augmentation, model building, and evaluation to achieve high accuracy in predictions. The final model can effectively classify new images as either containing a cat or a dog.
