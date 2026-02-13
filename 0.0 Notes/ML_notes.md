
# 📘 Machine Learning Notes

---

## 1. Machine Learning vs Deep Learning

- **Machine Learning (ML):**
  - Works on structured data (CSV, tabular).
  - Needs manual feature extraction (Feature Engineering).
  - Examples: Linear Regression, Decision Trees, SVM.

- **Deep Learning (DL):**
  - Works well with unstructured data (images, audio, text).
  - Learns features automatically (Feature Learning).
  - Uses Neural Networks (CNN, RNN, Transformers).

- **Key difference:** DL ⊂ ML but requires **large data + high compute power (GPU/TPU)**.

---

## 2. Linear Regression

### 🔹 Goal
- Fit a **best-fit line** between input (X) and output (Y).
- Equation (Hypothesis Function):  
  \[
  h_\theta(x) = \theta_0 + \theta_1 x
  \]

### 🔹 Cost Function
- Measures error between predicted and actual values.  
  **Mean Squared Error (MSE):**
  \[
  J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2
  \]

- **Graph of Cost Function**: Convex curve (U-shaped parabola).  
  - Global Minima = Best set of parameters (θ₀, θ₁).
  - Local Minima not a problem (MSE is convex).

### 🔹 Gradient Descent Algorithm
- Objective: Minimize cost by updating parameters.
- Update rule:  
  \[
  \theta_j = \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}
  \]
  where **α = Learning rate**.

- For Linear Regression:
  \[
  \theta_0 = \theta_0 - \alpha \cdot \frac{1}{m} \sum (h_\theta(x) - y)
  \]
  \[
  \theta_1 = \theta_1 - \alpha \cdot \frac{1}{m} \sum (h_\theta(x) - y) \cdot x
  \]

- **Learning Rate (α):**
  - Too small → Slow convergence.
  - Too large → Overshoots, may diverge.

---

## 3. Performance Metrics (Regression)

- **R² (Coefficient of Determination):**
  \[
  R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
  \]
  where  
  \( SS_{res} = \sum (y_i - \hat{y}_i)^2 \),  
  \( SS_{tot} = \sum (y_i - \bar{y})^2 \).

  - Value between **0 and 1**.  
  - Higher → Better fit.

- **Adjusted R²:**
  \[
  R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}
  \]
  where \( n \)=no. of samples, \( p \)=predictors.  
  - Penalizes unnecessary variables.  
  - Important in **multiple regression**.

---

## 4. Overfitting & Underfitting

- **Overfitting (High Variance, Low Bias):**
  - Performs well on training data but poorly on test data.
  - Example: Training Acc = 90%, Test Acc = 80%.

- **Underfitting (High Bias, High Variance):**
  - Performs poorly on both train & test.
  - Example: Train = 70%, Test = 65%.

- **Generalized Model (Low Bias, Low Variance):**
  - Train = 92%, Test = 91% (Best).

---

## 5. Ridge & Lasso Regression

### 🔹 Ridge Regression (L2 Regularization)
- Adds penalty to coefficients to prevent steep slope:
  \[
  J(\theta) = \frac{1}{2m} \sum (h_\theta(x)-y)^2 + \lambda \sum_{j=1}^p \theta_j^2
  \]
- Shrinks coefficients but never makes them zero.
- Good when **all features are useful**.

### 🔹 Lasso Regression (L1 Regularization)
- Penalty is absolute sum:
  \[
  J(\theta) = \frac{1}{2m} \sum (h_\theta(x)-y)^2 + \lambda \sum_{j=1}^p |\theta_j|
  \]
- Some coefficients become **zero → Feature Selection**.
- Useful when many features are irrelevant.

---

## 6. Assumptions of Linear Regression
1. **Linearity:** Relationship between X and Y must be linear.  
2. **Normality of Residuals:** Errors follow Gaussian distribution.  
3. **No Multicollinearity:** Independent variables should not be highly correlated.  
   - Variance Inflation Factor (VIF) used to check.  
4. **Homoscedasticity:** Equal variance of errors.  
5. **Feature Scaling:** Standardization (Z-score) helps faster convergence.

---

## 7. Logistic Regression (Classification)

### 🔹 Why not Linear Regression?
- Predictions may go beyond [0,1].
- Not suitable for classification.

### 🔹 Hypothesis (Sigmoid Function)
\[
h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}
\]
- Squashes output between 0 and 1.
- Threshold at 0.5 → classification.

### 🔹 Decision Boundary
- Defined where \( h_\theta(x) = 0.5 \).  
- Separates classes (Pass/Fail).

### 🔹 Cost Function
- Linear regression cost → Non-convex in classification.
- Logistic Regression Cost (Log Loss):  
  \[
  J(\theta) = - \frac{1}{m} \sum \Big[ y \log(h_\theta(x)) + (1-y)\log(1-h_\theta(x)) \Big]
  \]
- Convex → Gradient Descent finds **Global Minima**.

---

## 8. Performance Metrics (Classification)

### 🔹 Confusion Matrix
|               | Predicted 1 | Predicted 0 |
|---------------|-------------|-------------|
| Actual 1      | TP          | FN          |
| Actual 0      | FP          | TN          |

- **Accuracy:**  
  \[
  \frac{TP+TN}{TP+FP+TN+FN}
  \]

- **Precision (Positive Predictive Value):**  
  \[
  \frac{TP}{TP+FP}
  \]
  - High precision → Few false positives.  
  - Example: Spam Detection.

- **Recall (Sensitivity):**  
  \[
  \frac{TP}{TP+FN}
  \]
  - High recall → Few false negatives.  
  - Example: Cancer Detection.

- **F1 Score:**  
  \[
  F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
  \]

- **Fβ Score:**  
  \[
  F_\beta = \frac{(1+\beta^2) \cdot Precision \cdot Recall}{\beta^2 \cdot Precision + Recall}
  \]
  - β > 1 → Recall priority.  
  - β < 1 → Precision priority.

---

## 🔑 Interview Tips
- Be clear on **bias-variance tradeoff** with examples.  
- Know difference between **R² vs Adjusted R²**.  
- Be able to derive/update rules for **gradient descent**.  
- Explain **why logistic regression uses log-loss** (convexity).  
- Know practical cases:  
  - Spam Classification → Precision focus.  
  - Cancer Detection → Recall focus.  
- Ridge vs Lasso: Ridge = shrinkage, Lasso = shrinkage + feature selection.

---
