# **Credit Risk Analysis & Prediction**

## **Overview**
This project focuses on developing a machine learning and deep learning model to predict customer credit scores based on various financial and demographic features. The model classifies customers into "Good," "Standard," and "Poor" credit categories, aiding in effective credit risk management.

## **Objective**
- Analyze consumer and commercial credit portfolios and identify key factors influencing credit scores.
- Build an **Artificial Neural Network (ANN)** for accurate credit score classification.
- Optimize data preprocessing and feature engineering techniques for improved model performance.

## **Dataset**
The dataset was sourced from **Kaggle's "Credit Score Classification"** dataset and includes:
- **Demographic Information**: Age, Occupation, Income, etc.
- **Financial Indicators**: Number of bank accounts, credit card count, loan history, outstanding debt.
- **Payment Behavior**: EMI per month, delayed payments, credit utilization ratio.
- **Target Variable**: Credit Score (**Good, Standard, Poor**)

## **Data Preprocessing & Feature Engineering**
- **Data Cleaning**: Removed null values, special characters, and redundant columns.
- **Feature Engineering**:
  - Converted categorical variables using **One-Hot Encoding** and **Ordinal Encoding**.
  - Transformed "Credit_History_Age" into months.
  - Removed outliers using **Interquartile Range (IQR) Method**.
- **Handling Class Imbalance**:
  - Used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset.
- **Feature Scaling**:
  - Applied **MinMax Scaling** to normalize numerical features.

## **Model Implementation**
### **Machine Learning & Deep Learning Approach**
- **Model Architecture**:
  - Input Layer with feature transformations.
  - Multiple **Dense Layers** with **Batch Normalization** and **Dropout Regularization**.
  - Final **Softmax Layer** for multi-class classification.
- **Optimization Strategy**:
  - Loss Function: `sparse_categorical_crossentropy`
  - Optimizer: `Adam` (learning rate = 0.001)
  - Early Stopping: Stops training if validation accuracy doesn’t improve for 30 epochs.

## **Performance Metrics**
- **Validation Accuracy**: **74%**
- **Precision-Recall Analysis**:
  - Good Credit: **76% F1-score**
  - Standard Credit: **68% F1-score**
  - Poor Credit: **73% F1-score**
- **Confusion Matrix**: Evaluated misclassifications and optimized decision boundaries.

## **Key Findings & Improvements**
✅ Balanced dataset using **SMOTE**, reducing class imbalance.
✅ Implemented **feature engineering and data preprocessing** to improve accuracy.
✅ Utilized **early stopping and dropout layers** to prevent overfitting.

**Future Improvements:**
- Fine-tune hyperparameters using **GridSearchCV**.
- Explore alternative models like **XGBoost, LightGBM** for improved generalization.
- Implement **Explainable AI (XAI)** techniques to interpret model predictions.

## **Technology Stack**
- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-Learn, TensorFlow, Keras, Matplotlib, Seaborn
- **Data Handling**: SMOTE, MinMax Scaling, One-Hot & Ordinal Encoding
- **Model Deployment**: TensorFlow & Keras ANN

## **How to Run the Project**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/credit-risk-analysis.git
   cd credit-risk-analysis
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook credit_risk_analysis.ipynb
   ```



