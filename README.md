# Fraud Detection Model

## Overview
This project aims to build a fraud detection model using machine learning techniques. The dataset used in this project contains transaction data with various features such as transaction amount, device type, user age, income, debt, and credit score. The goal is to predict whether a transaction is fraudulent or not based on these features.

## Dataset
The dataset used in this project is provided in the file `fraud.csv`. It includes the following columns:

- `user_id`: Unique identifier for each user
- `amount`: Transaction amount
- `device_type`: Type of device used for the transaction (categorical)
- `is_fraud`: Binary target variable indicating whether the transaction is fraudulent (0: not fraudulent, 1: fraudulent)
- `age`: User age
- `income`: User income
- `debt`: User debt
- `credit_score`: User credit score

## Preprocessing
- Missing Values: No missing values were found in the dataset.
- Outliers: Outliers were checked using boxplot visualization for numerical features, and no outliers were detected.
- Scaling: Numerical features were scaled using StandardScaler to standardize the distribution.
- Encoding: The categorical feature `device_type` was encoded using LabelEncoder to convert it into numeric format.

## Modeling
- Neural Network Architecture: A simple feedforward neural network model was built using TensorFlow's Keras API. The model consists of three dense layers with ReLU activation functions.
- Compilation: The model was compiled with the Adam optimizer and binary cross-entropy loss function.
- Training: The model was trained on the training data for 2 epochs with a batch size of 32.

## Evaluation
- Evaluation Metrics: The model's performance was evaluated using several metrics including Recall Score, F1 Score, Accuracy Score, and ROC-AUC Score.
- Results: The trained model achieved satisfactory performance on the test data with the following evaluation metrics:
  - Recall Score: 99.78 %
  - F1 Score: 99.79 %
  - Accuracy Score: 99.79 %
  - ROC-AUC Score: 99.99 %
