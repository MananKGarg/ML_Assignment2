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

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9197 | 0.9713 | 0.9317 | 0.8744 | 0.9021 | 0.8353 |
| **Decision Tree** | 0.9197 | 0.9189 | 0.9202 | 0.8872 | 0.9034 | 0.8351 |
| **kNN** | 0.8936 | 0.9452 | 0.8989 | 0.8436 | 0.8704 | 0.7814 |
| **Naive Bayes** | 0.8219 | 0.9263 | 0.7233 | 0.9385 | 0.8170 | 0.6701 |
| **Random Forest (Ensemble)** | 0.9555 | 0.9852 | 0.9755 | 0.9179 | 0.9458 | 0.9093 |
| **XGBoost (Ensemble)** | 0.9566 | 0.9883 | 0.9730 | 0.9231 | 0.9474 | 0.9114 |

**Observations on Model Performance:**

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed strongly as a baseline, tying with the Decision Tree for accuracy (0.9197) but achieving a significantly higher AUC (0.9713 vs 0.9189). This indicates it separates the spam/non-spam probabilities better overall, though its recall is slightly lacking. |
| **Decision Tree** | Achieved identical accuracy to Logistic Regression but with a better F1 score (0.9034). However, it yielded the lowest AUC of all models (0.9189), suggesting it made hard splits that didn't scale well probabilistically across the test data. |
| **KNN** | Performed adequately (0.8936 accuracy) but was the second-lowest overall. The high dimensionality of the Spambase dataset (57 features) likely triggered the "curse of dimensionality," making distance-based calculations less effective. |
| **Naive Bayes** | Yielded the lowest overall accuracy, precision, F1, and MCC. Interestingly, it had the highest recall of any model (0.9385). It aggressively classified emails as spam, catching almost all of them, but made a massive amount of false-positive mistakes along the way. |
| **Random Forest (Ensemble)** | An excellent performer with 0.9555 accuracy. It achieved the highest precision of all models (0.9755), meaning when it flags an email as spam, it is almost certainly correct, minimizing annoying false positives for users. |
| **XGBoost (Ensemble)** | The absolute best-performing model across almost all metrics, leading in Accuracy (0.9566), AUC (0.9883), F1 (0.9474), and MCC (0.9114). Its sequential boosting handled the continuous frequency features exceptionally well, proving to be the most robust choice. |