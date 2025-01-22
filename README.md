# Credit Score Classification Projec
This project aims to classify a user's credit score. Various data preprocessing steps have been applied to the dataset, imbalance issues have been addressed, and different machine learning models have been used for classification.

Project Content
Data Loading and Exploration

The dataset was loaded from train.csv.
The structure of the dataset was analyzed using methods like info() and isnull().sum().
Missing values were checked.
Data Visualization and Analysis

The class distribution of credit scores was visualized using a bar chart.
A correlation matrix of numerical features was created and displayed as a heatmap.
Histograms were drawn to examine the distribution of features.
Encoding the Target Variable

The categorical target variable (Credit_Score) was converted to numeric labels.
Addressing Data Imbalance (SMOTE)

Imbalanced classes in the dataset were balanced using the SMOTE (Synthetic Minority Oversampling Technique) method.
Feature Selection and Scaling

Features likely to be meaningful for the model were selected.
Selected features were scaled using StandardScaler.
Machine Learning Models and Optimization

Algorithms such as RandomForestClassifier, AdaBoostClassifier, XGBClassifier, and LGBMClassifier were utilized.
Hyperparameter optimization was performed using RandomizedSearchCV to select the best model.
Model Evaluation

The best model's accuracy score, classification report, confusion matrix, and predicted class distributions were examined.
A learning curve was used to visualize the model's training and validation performance.
Feature Importance Analysis

The feature importance scores of the selected model were visualized to identify the most impactful features.
Requirements
The following Python libraries were used to run this project:

pandas
numpy
seaborn
matplotlib
scikit-learn
xgboost
lightgbm
imblearn
To install the necessary libraries, you can use the following command:
pip install pandas numpy seaborn matplotlib scikit-learn xgboost lightgbm imbalanced-learn
