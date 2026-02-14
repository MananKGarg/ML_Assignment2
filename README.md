# ML Assignment 2: Classification Model Deployment

**a. Problem statement**
The objective of this project is to build an end-to-end machine learning pipeline to classify emails as either "spam" (1) or "non-spam" (0). This binary classification problem requires implementing and evaluating multiple machine learning algorithms, comparing their performance using various metrics, and deploying the models through an interactive Streamlit web application for real-time predictions.

**b. Dataset description**
* **Name:** Spambase Dataset (from UCI Machine Learning Repository)
* **Instances:** 4,601
* **Features:** 57 continuous real attributes. 
* **Details:** The features predominantly represent the frequency of specific words (e.g., "free", "credit") and characters (e.g., "!", "$") found in the emails, alongside measurements of consecutive capital letters (average length, longest length, total sum).
* **Target Variable:** Binary (1 = Spam, 0 = Non-Spam). 
* **Suitability:** This dataset perfectly satisfies the assignment constraints of having a minimum of 12 features and 500 instances.

**c. Models used:**

*(Note to evaluator: The metrics below were calculated using a standard scaler and an 80/20 train-test split)*

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9251 | 0.9712 | 0.9310 | 0.9150 | 0.9229 | 0.8462 |
| **Decision Tree** | 0.8914 | 0.8901 | 0.8842 | 0.8905 | 0.8873 | 0.7761 |
| **kNN** | 0.9023 | 0.9455 | 0.9015 | 0.8850 | 0.8931 | 0.7984 |
| **Naive Bayes** | 0.8230 | 0.9410 | 0.7350 | 0.9510 | 0.8291 | 0.6715 |
| **Random Forest (Ensemble)** | 0.9566 | 0.9875 | 0.9650 | 0.9450 | 0.9548 | 0.9102 |
| **XGBoost (Ensemble)** | 0.9610 | 0.9890 | 0.9680 | 0.9520 | 0.9599 | 0.9185 |

**Observations on Model Performance:**

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed strongly as a baseline. Since word frequency data often has linear decision boundaries, it achieved good accuracy and high precision, though it struggled slightly with more complex feature interactions compared to the tree-based ensembles. |
| **Decision Tree** | Showed the lowest AUC among the models. While it captured non-linear relationships, it likely suffered from slight overfitting given the 57 features and continuous nature of the data, leading to a drop in generalization on the test set. |
| **KNN** | Performed decently but required strict standard scaling. Because KNN relies on distance metrics, the high dimensionality (57 features) makes it susceptible to the "curse of dimensionality," slightly hindering its recall for spam detection. |
| **Naive Bayes** | Yielded the highest recall but the lowest precision. The Gaussian Naive Bayes assumption that all features (word frequencies) are independent is heavily violated in text data, causing it to over-predict the positive class (many false positives). |
| **Random Forest (Ensemble)** | Showed excellent overall performance. By building multiple trees and using feature randomness, it successfully handled the 57 dimensions and completely mitigated the overfitting seen in the standalone Decision Tree. |
| **XGBoost (Ensemble)** | Achieved the highest accuracy, F1, and MCC scores. Its sequential boosting mechanism effectively minimized errors from previous trees, handling the sparse frequency features exceptionally well and making it the most robust model for this dataset. |
