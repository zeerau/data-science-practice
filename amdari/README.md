
# Customer Churn Prediction

## Overview

This project aims to predict customer churn based on various customer attributes and behaviors. By analyzing factors such as demographics, purchase history, website usage, and engagement metrics, this project develops machine learning models to identify customers who are likely to churn.

## Processes

The project follows these steps:

1.  **Data Loading and Exploration**: The dataset, containing customer information and their churn status, is loaded and explored to understand its structure, identify missing values, and analyze basic statistics and distributions of key features.
2.  **Feature Engineering**: Relevant features are extracted from nested columns in the dataset, such as purchase frequency, subscription duration, website activity, and engagement metrics. Categorical features like Gender, Engagement Frequency, and Subscription Plan are encoded into numerical representations.
3.  **Correlation Analysis**: The correlation between different features and the churn label is analyzed to understand which factors are most strongly associated with churn.
4.  **Data Splitting**: The dataset is split into training and testing sets to train and evaluate the performance of machine learning models.
5.  **Data Scaling**: Numerical features are scaled using StandardScaler to ensure that no single feature dominates the model training process.
6.  **Model Training and Evaluation**: Logistic Regression and Decision Tree Classifier models are trained on the preprocessed data. Their performance is evaluated using metrics such as accuracy, precision, recall, and F1-score on both the training and testing sets.
7.  **Confusion Matrix Analysis**: Confusion matrices are plotted for both models to visualize their performance in classifying churn and non-churn customers.

## Tools

The following tools and libraries were used in this project:

*   **pandas**: For data manipulation and analysis.
*   **numpy**: For numerical operations.
*   **matplotlib.pyplot** and **seaborn**: For data visualization.
*   **sklearn**: For machine learning model training and evaluation, including `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `ConfusionMatrixDisplay`, `train_test_split`, `StandardScaler`, `LogisticRegression`, and `DecisionTreeClassifier`.
*   **tqdm**: For displaying progress bars (although not explicitly used in the provided code, it's imported).
*   **ast**: For safely evaluating string literals containing Python expressions.
*   **warnings**: For managing warnings.

## Conclusion

The project successfully demonstrates the process of predicting customer churn using machine learning. By extracting relevant features, preprocessing the data, and training models like Logistic Regression and Decision Tree Classifier, we can identify potential churners. The evaluation metrics and confusion matrices provide insights into the performance of these models. Further steps could involve exploring other machine learning algorithms, hyperparameter tuning, and feature selection to potentially improve the prediction accuracy.
