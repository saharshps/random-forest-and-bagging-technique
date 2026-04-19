# Sales Prediction using Bagging & Random Forest

## Overview

This project predicts whether sales are high or not using machine learning models like Bagging and Random Forest.

## Dataset

* File: `sales.csv`
* Target variable: `high`

## Steps

1. Load and explore the dataset
2. Encode categorical variables using LabelEncoder
3. Split data into training and testing sets
4. Train models:

   * Bagging Classifier
   * Random Forest Classifier
5. Tune Random Forest using GridSearchCV
6. Evaluate using:

   * Accuracy
   * Confusion Matrix
   * Cross-validation
   * ROC Curve

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python your_script_name.py
```

## Result

* Random Forest performs better than Bagging
* Model performance evaluated using multiple metrics

## Author

Sai Saharsh
