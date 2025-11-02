# Machine_Learning_Project
Credit Card Fraud Detection using Machine Learning

## **1. Project Overview**

Fraud with credit cards turned into one of the most acute financial crimes in the digital economy. In real time, it is a significant challenge because the imbalance between classes (very few cases of fraud compared to the authentic ones) is enormous.

The proposed project focuses on creating a Fraud Detection System developed usingMachine Learning that subsequently detects potentially fraudulent transactions directly on historical information and reports them to help keep financial losses at a minimal and secure transaction.

---

## **2. Project Objectives**

* To analyze credit card transaction data and identify patterns of fraud.
* To handle class imbalance effectively using resampling or algorithmic strategies.
* To train and evaluate multiple machine learning models for fraud detection.
* To compare model performance using suitable evaluation metrics.
* To build a deployable and scalable pipeline for predictive analytics.

---

## **3. Dataset Information**

### **Dataset Details**

* **Number of records:** 284,807
* **Number of features:** 30 (including target variable)
* **Fraudulent transactions:** 492 (≈ 0.17%)
* **Non-fraudulent transactions:** 284,315

### **Attributes Description**

| Feature    | Description                                                                        |
| ---------- | ---------------------------------------------------------------------------------- |
| `Time`     | Seconds elapsed between this transaction and the first transaction in the dataset. |
| `V1 – V28` | PCA-transformed numerical features (actual features are confidential).             |
| `Amount`   | Transaction amount (useful for scaling).                                           |
| `Class`    | Target variable — 1 indicates fraud, 0 indicates genuine transaction.              |

---

## **4. Tools and Libraries Used**

* **Language:** Python 3
* **Libraries:**

  * Data Handling: `pandas`, `numpy`
  * Visualization: `matplotlib`, `seaborn`
  * Machine Learning: `scikit-learn`, `xgboost`
  * Evaluation Metrics: `classification_report`, `roc_auc_score`, `confusion_matrix`

---

## **5. Methodology**

### **Step 1: Data Preprocessing**

* Load the dataset using `pandas`.
* Check for missing values (none found).
* Scale `Time` and `Amount` using `StandardScaler`.
* Split the dataset into **80% training** and **20% testing** subsets.

### **Step 2: Exploratory Data Analysis (EDA)**

* Visualize class imbalance using bar plots.
* Observe that only 0.17% of transactions were fraudulent.
* Explore correlations among PCA features.

### **Step 3: Handling Class Imbalance**

* Implemented **SMOTE (Synthetic Minority Oversampling Technique)** to generate synthetic samples of the minority class.
* Alternatively, used `class_weight='balanced'` in Logistic Regression and Random Forest.

### **Step 4: Model Building**

* **Logistic Regression** — simple, interpretable baseline.
* **Random Forest Classifier** — ensemble method providing strong accuracy.
* **XGBoost Classifier** — advanced boosting model for optimized performance.

### **Step 5: Model Evaluation**

Used **classification metrics** suited for imbalanced datasets:

* **Accuracy:** Overall correctness of predictions.
* **Precision:** Correctly identified frauds among predicted frauds.
* **Recall:** Correctly identified frauds among actual frauds (very important).
* **F1-Score:** Balance between precision and recall.
* **ROC-AUC:** Measures model discrimination capability.

---

## **6. Results and Performance**
<img width="728" height="631" alt="image" src="https://github.com/user-attachments/assets/8ffe7188-8a45-4d77-a2c6-29f6d92a66cb" />


---

## **7. Visualizations**

* **Class Distribution:** Displays severe class imbalance.
* **Confusion Matrix:** Visualizes true vs predicted classes.
* **ROC Curve:** Evaluates performance using TPR vs FPR.
* **Feature Importance (Random Forest / XGBoost):** Shows top features influencing predictions.

---

## **8. Key Insights**

* The data is very imbalanced, they need specific approaches to handling them.
* **Recall** gains precedence over accuracy - we would like to identify as many frauds as possible.
Random Forest and XGBoost models are very effective in comparison to Logistic Regression.
PCA transformed features mask meaningfulness about the real world, but still can be used to predict patterns.
* Convergence is improved when there is feature scaling (particularly on the features of Amount and Time).

---

## **9. Limitations**

* features are anonymized (V1 -V28), thus restricting business interpretation.
* The model performance is dependent on data distribution; new ways of committing fraud may not be detected.
* 2013 (not necessarily a reputable source of data on current-day fraud trends).

---

## **10. Future Enhancements**

* Use **deep learning models (LSTM/Autoencoders)** for anomaly detection.
* Implement **real-time fraud detection** using streaming data.
* Explore **explainable AI (SHAP/LIME)** for interpretability.
* Integrate model into a **web or mobile dashboard** for live prediction.

---

## **11. Conclusion**

This project has been able to showcase machine learning algorithms applied in detecting credit card frauds.
The model training and evaluation involved preprocessing, training the models, and testing on ensemble models were highly accurate with a good recall on the highest accuracy in both the Random Forest and XGBoost models.

---

## **12. References**

1. Kaggle Dataset — [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Dal Pozzolo et al. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification.*
3. Scikit-learn Documentation — [https://scikit-learn.org](https://scikit-learn.org)
4. XGBoost Documentation — [https://xgboost.readthedocs.io](https://xgboost.readthedocs.io)

---

## **Project Team (Example)**

| Name              | PRN            | Role                           |
| ----------------- | -------------- | ------------------------------ |
| Pritam Morey      | 22070521134    | Student                        |
| **Guide:**        | Dr. Deepak Asudani Sir | Project Mentor         |
