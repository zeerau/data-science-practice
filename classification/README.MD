# Decision Trees for Credit Risk Classification

## Introduction
This project focuses on building a decision tree model to classify credit risk using the Statlog (German Credit Data) dataset. The dataset contains information about individuals described by a set of attributes and classifies them as good or bad credit risks. The goal of this project is to preprocess the data, train a decision tree classifier, and evaluate its performance.

## Process
1. **Dataset Acquisition**:  
   The dataset was fetched from the UCI Machine Learning Repository using the `ucimlrepo` library.

2. **Data Exploration**:  
   - Inspected the dataset's metadata and variable information.
   - Displayed the first few rows of the dataset to understand its structure.

3. **Data Preprocessing**:  
   - Replaced categorical codes with descriptive labels for better interpretability.
   - Converted categorical features into one-hot encoded variables using `pandas.get_dummies`.
   - Transformed the target variable (`y`) into a boolean format (`True` for good credit risk and `False` for bad credit risk).

4. **Model Training**:  
   - Split the dataset into training and testing sets using an 80-20 split.
   - Trained a decision tree classifier using the `sklearn.tree.DecisionTreeClassifier`.

5. **Model Evaluation**:  
   - Evaluated the model's performance using a confusion matrix and classification report.

## Tech Tools
- **Python**: Programming language used for data processing and model building.
- **Libraries**:
  - `ucimlrepo`: For fetching the dataset.
  - `pandas` and `numpy`: For data manipulation and preprocessing.
  - `matplotlib` and `seaborn`: For data visualization.
  - `scikit-learn`: For machine learning model training and evaluation.

## Conclusion
The decision tree model achieved a reasonable accuracy in classifying credit risks. The confusion matrix and classification report provided insights into the model's precision, recall, and F1-score. This project demonstrates the effectiveness of decision trees in handling classification problems with categorical and numerical data.

## Tags
`Decision Trees` `Credit Risk Classification` `Machine Learning` `Python` `Data Science` `UCI Dataset` `Statlog German Credit Data`