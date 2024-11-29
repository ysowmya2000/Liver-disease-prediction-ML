# Liver-disease-prediction-ML
This project leverages multiple machine learning models to predict the presence of liver disease based on clinical parameters. It includes preprocessing, exploratory data analysis, and comparison of various classification models to achieve optimal performance.

Liver diseases are among the most significant health concerns globally. This project aims to predict liver disease using a dataset of clinical parameters. Various machine learning models are implemented to classify the presence or absence of liver disease effectively.

## Project Workflow
### 1.Data Preprocessing
- Removal of duplicate and missing values.
- Handling of categorical values (e.g., converting 'Gender' to numeric).
- Standardization of feature values.

### 2. Exploratory Data Analysis (EDA)
- Visualization of feature distributions.
- Correlation analysis using heatmaps.

### 3. Model Implementation
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

### 4. Model Evaluation
- Accuracy, confusion matrix, and classification report.
- ROC curves and AUC scores.
- Hyperparameter tuning using GridSearchCV.

### 5. Performance Comparison
- Bar plots of accuracy and ROC scores for all models.

## Models Used
- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

## Performance Evaluation
- Metrics:
  - Accuracy
  - ROC-AUC
  - Precision, Recall, F1-Score
## Visualizations:
- ROC curves for all models.
- Bar plots comparing accuracy and ROC-AUC scores.

### Requirements
The following Python libraries are required:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- pickle

The best performance model is Logistic Regression
- Accuracy: 77.8%
- ROC:65.66

### Dataset
https://www.kaggle.com/datasets/uciml/indian-liver-patient-records
