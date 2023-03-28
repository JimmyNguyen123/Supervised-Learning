# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
The analysis aims to use Logistic Regression to predict whether a loan is healthy (0) or high-risk (1) by dividing the dataset to: training and testing data. The model use training data (X_train, y_train) as learning tool then use X_test to make prediction to y. The computed y will be compared with y_test to determine model's effectiveness through precision, recall, f-1 score, accuracy metrics.

* Explain what financial information the data was on, and what you needed to predict.
Variable need to predict is loan_status
Financial data available:
    loan_size
    interest_rate
    borrower_income
    debt_to_income
    num_of_accounts
    derogatory_marks
    total_debt

* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
Healthy loans (0): 75,036
High-risk loans (1): 2,500

* Describe the stages of the machine learning process you went through as part of this analysis.
    Sclale data to standardized all X variables
    Split data to training dataset (X_train,y_train) and testing dataset (X_test, y_test)
    Fit the Logistic regression model using X_train and y_train
    After training model, predict y using X_test
    Compare y predicted with y_test to evaluate model's accuracy, precision, recall, f-1 score
    Restructure the training data by using resampling method on X_train and y_train
    Fit the model by using resampled data (X_resampled, y_resampled) then predict y
    Compare new y predicted to y_test
    Compare 2 model to evaluate performance

* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
LogisticRegression: using available financial data as independent variables to predict loan_status (Dependent variable). End result is either 0 or 1

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.

                precision    recall  f1-score   support

  Heathly Loan       1.00      1.00      1.00     18759
High-risk Loan       0.87      0.98      0.92       625

      accuracy                           0.99     19384
     macro avg       0.94      0.99      0.96     19384
  weighted avg       1.00      0.99      0.99     19384
  
Precision: Healthy loan's precision of 1.00 indicates 100% of loans that the model predict as healthy are actually healthy. Similarly, 87% of predicted high-risk loans are actually high-risk. 
Recall: the model 100% correctly identifies healthy loans and 98% with high-risk ones
F1-score: The model performs better in predicting Healthy loans than high-risk loans
Accuracy: The model predict 99% correctly the test data
Macro avg: the unweighted average of precision and recall, this means the model has higher recall rate than precision rate. Overall performance is 0.96 (f-1 score), a very high score.
Weighted avg: the model has high weighted avg meaning its performance is not skewed by the category imbalance

* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
  
                precision    recall  f1-score   support

  Heathly Loan       1.00      0.99      1.00     18759
High-risk Loan       0.87      1.00      0.93       625

      accuracy                           0.99     19384
     macro avg       0.93      1.00      0.96     19384
  weighted avg       1.00      0.99      1.00     19384
  
Precision: Same as model 1. 100% cases of healthy loans predicted by model 2 are actually healthy loans. Similarily, 87% for high-risk are actually high-risk
Recall: Model 2 yields higher recall rate in high-risk loan which means model 2 produce less FN case for high-risk loans than model 1. Also, Model 2 has lower recall rate in Healthy-loan meaning it fail to identify the 1% healthy loans
F1-score: same as model 1, model 2 performs better in terms of predicting healthy loan than high-risk loan
Accuracy: Model 2 predict 99% correctly the test data
Macro avg: model 2 has higher recall rate than precision
Weighted avg: model's performance is not affected by class imbalance

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
In my opinion, model 2 will be more reliable to use than model 1 due to lower recall rate in High-risk loan. 

* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
Yes, assuming the primary goal is that we want to minimize situations where a loan is actually a high-risk loan but we identify it as a healthy-loan. The bank may end up issuing bad loans that have high possibility to be default. This means we are trying to have a model that produces lower False Negative cases in high-risk loan, which equivalent to higher recall rate.

