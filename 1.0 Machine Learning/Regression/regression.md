Let’s break it **step by step** for **Linear, Ridge, and Lasso** so you see **what is happening under the hood**.  

---

## 🔹 1. Dataset Preparation
- We generated **synthetic data** (`make_regression`) with:
  - **500 samples** (rows, like houses/observations).  
  - **5 features** (`feature_0` … `feature_4`).  
  - Target variable = `price`.  
- **Train-Test Split:**
  - **80% Train** (400 samples).  
  - **20% Test** (100 samples).  

---

## 🔹 2. Model Training & Testing

### **(a) Linear Regression**
- **What it does:**  
  Finds the best straight line (hyperplane in multi-feature case) that minimizes **Mean Squared Error (MSE)** without any penalty on coefficients.  
  Equation:  
  \[
  y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n
  \]

- **Process:**  
  1. Train on **X_train, y_train** → fits coefficients (weights).  
  2. Predict on **X_test**.  
  3. Evaluate with **MSE**.  

- **Result (your case):**  
  - **Train:** Fits data well but may overfit.  
  - **Test MSE:** 425.30  

---

### **(b) Ridge Regression**
- **What it does:**  
  Same as Linear Regression **+ penalty on large coefficients**.  
  Equation adds **L2 regularization**:  
  \[
  \text{Loss} = \text{MSE} + \alpha \sum \beta_j^2
  \]
  - Prevents coefficients from becoming very large.  
  - Helps when features are correlated (multicollinearity).  

- **Process:**  
  1. Train with penalty on weights (shrinks but does not eliminate them).  
  2. Predict on **X_test**.  
  3. Evaluate with MSE.  

- **Result (your case):**  
  - **Test MSE:** 424.50 (slightly better than Linear).  
  - **CV MSE:** 381.14 (stable performance).  

---

### **(c) Lasso Regression**
- **What it does:**  
  Same as Linear Regression **+ penalty on absolute values of coefficients**.  
  Equation adds **L1 regularization**:  
  \[
  \text{Loss} = \text{MSE} + \alpha \sum |\beta_j|
  \]
  - Shrinks some coefficients exactly to **0** → does **feature selection**.  

- **Process:**  
  1. Train with L1 penalty.  
  2. Some features may be dropped (coefficients = 0).  
  3. Predict on **X_test**.  
  4. Evaluate with MSE.  

- **Result (your case):**  
  - **Test MSE:** 424.58 (very close to Ridge).  
  - **CV MSE:** 381.11 (slightly better than Ridge).  

---

## 🔹 3. Major Function of Each Model
- **Linear Regression:** Fits the line → no control on coefficient size. Risk of overfitting.  
- **Ridge Regression:** Controls coefficient size with L2 penalty → improves stability.  
- **Lasso Regression:** Shrinks some coefficients to 0 → performs feature selection + prevents overfitting.  

---

## 🔹 4. Final Verdict
- **Linear Regression:** Simple baseline model.  
- **Ridge:** Better generalization by shrinking weights.  
- **Lasso:** Better when you want both prediction + automatic feature selection.  
- **In your dataset:** All three are close, but **Ridge and Lasso are slightly better** than Linear.  

---
