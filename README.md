# ❤️ Heart Disease Prediction using Machine Learning & Deep Learning

## 📌 Project Overview

Cardiovascular diseases are among the leading causes of mortality
worldwide. Early detection plays a crucial role in improving survival
rates.

This project builds and compares multiple Machine Learning and Deep
Learning models to predict the presence of heart disease using patient
clinical data.

The objective is to evaluate model performance using multiple evaluation
metrics and determine the most effective predictive approach.

------------------------------------------------------------------------

## 📊 Dataset Description

The dataset contains the following medical attributes:

  Feature    Description
  ---------- -----------------------------------
  age        Age of patient
  sex        Gender (0 = female, 1 = male)
  cp         Chest pain type
  trestbps   Resting blood pressure
  chol       Serum cholesterol
  fbs        Fasting blood sugar
  restecg    Resting ECG results
  thalach    Maximum heart rate achieved
  exang      Exercise induced angina
  oldpeak    ST depression
  slope      Slope of peak exercise ST segment
  ca         Number of major vessels
  thal       Thalassemia
  num        Target variable

### 🎯 Target Encoding

-   0 → No heart disease\
-   1 → Heart disease present

The original target values were binarized to convert the problem into
binary classification.

------------------------------------------------------------------------

## ⚙️ Data Preprocessing

To ensure clean and reproducible processing, a ColumnTransformer
pipeline was used.

### 🔢 Numerical Features

-   Missing values handled using Mean Imputation
-   Standardized using StandardScaler

### 🔤 Categorical Features

-   Missing values handled using Most Frequent Imputation
-   Encoded using OneHotEncoder

### 📂 Train-Test Split

-   80% Training\
-   20% Testing\
-   Stratified to preserve class balance

------------------------------------------------------------------------

## 🤖 Models Implemented

The following models were implemented and tuned using GridSearchCV
(5-fold cross-validation):

### 1️⃣ Logistic Regression

-   Hyperparameter tuned: C

### 2️⃣ K-Nearest Neighbors (KNN)

-   Hyperparameter tuned: n_neighbors

### 3️⃣ Support Vector Machine (SVM)

-   Hyperparameters tuned: C, gamma

### 4️⃣ XGBoost

-   Hyperparameters tuned: max_depth, learning_rate

### 5️⃣ LightGBM

-   Hyperparameters tuned: max_depth, learning_rate

### 6️⃣ Multi-Layer Perceptron (PyTorch)

Neural Network Architecture: - Input Layer - Hidden Layer (64 neurons,
ReLU) - Hidden Layer (32 neurons, ReLU) - Output Layer (Sigmoid)

Training Details: - Loss Function: Binary Cross Entropy - Optimizer:
Adam - Epochs: 20 - Batch Size: 32

------------------------------------------------------------------------

## 📈 Evaluation Metrics

Each model was evaluated using:

-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   AUC-ROC
-   Confusion Matrix

These metrics provide a comprehensive understanding of classification
performance.

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   XGBoost
-   LightGBM
-   PyTorch

------------------------------------------------------------------------

## 📁 Project Structure

    heart-disease-prediction/
    │
    ├── heart_disease_notebook.ipynb
    ├── cleaned_heart_disease.csv
    ├── final_model_results.csv
    ├── model_comparison.png
    ├── presentation.pptx
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 🚀 How to Run This Project

### 1️⃣ Clone the repository

    git clone https://github.com/yourusername/heart-disease-prediction.git

### 2️⃣ Install dependencies

    pip install -r requirements.txt

### 3️⃣ Run the notebook

    jupyter notebook heart_disease_notebook.ipynb

------------------------------------------------------------------------

## 👨‍💻 Author

Jitendra Singh Kushwah
