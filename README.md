# deep-learning-challenge

#Alphabet Soup Funding Predictor

#Overview
The Alphabet Soup Funding Predictor project involves building a neural network model to predict the success of funding applicants. The project consists of data preprocessing, model creation and evaluation, optimization, and documentation. The goal is to classify whether an applicant's request for funding will be successful based on various features.

#Data Preprocessing

1. Load and Prepare Data
Data Source: charity_data.csv from the provided URL.
Target Variable: IS_SUCCESSFUL (binary classification).
Feature Variables: All other columns excluding EIN and NAME.

2. Data Cleaning

#Drop Columns: 

Removed EIN and NAME columns as they are not relevant for modeling.

#Handle Categorical Variables:

APPLICATION_TYPE and CLASSIFICATION were cleaned by grouping rare categories into an "Other" category.
Encode Categorical Data: Converted categorical variables to numerical values using one-hot encoding.
Feature and Target Arrays: Created feature array X and target array y.
Split Data: Divided the data into training and testing datasets.
Scale Data: Standardized the data using StandardScaler.

#Model Development

1. Define Neural Network
Model Architecture:
Input Layer: Number of neurons equal to the number of features.
Hidden Layers: Two hidden layers with 128 and 64 neurons respectively, using ReLU activation.
Output Layer: Single neuron with a sigmoid activation function for binary classification.

#2. Compile and Train
Compile: Used binary cross-entropy loss, Adam optimizer, and accuracy as the metric.
Train: Trained the model for 100 epochs with a batch size of 32 and included validation.

#3. Evaluate Model
Evaluation: Assessed model performance on the test data, checking both loss and accuracy.
Export Model: Saved the model to an HDF5 file named AlphabetSoupCharity.h5.

#Model Optimization

1. Re-Preprocessing
Repeated data preprocessing steps in a new notebook to ensure consistency.

2. Optimization Techniques
Hyperparameter Tuning: Implemented methods such as varying the number of neurons, layers, and activation functions.
Early Stopping: Used early stopping to prevent overfitting.
Regularization: Applied dropout and L2 regularization techniques.

3. Save Optimized Model
Exported the optimized model to an HDF5 file named AlphabetSoupCharity_Optimization.h5.

#Report

1. Analysis Overview
Purpose: To build and evaluate a neural network model to predict funding success based on the provided dataset.

2. Results
Target Variable: IS_SUCCESSFUL.
Feature Variables: All columns except EIN and NAME.
Columns Removed: EIN and NAME.
Model Architecture: Input layer, two hidden layers with ReLU activation, and an output layer with a sigmoid activation.
Performance: Achieved target performance metrics.
Optimization Steps: Hyperparameter tuning, early stopping, and regularization.

3. Summary
The neural network model successfully predicts funding success, with detailed results provided in the evaluation section. An alternative model suggestion includes using ensemble methods like Random Forests for potentially improved performance.

