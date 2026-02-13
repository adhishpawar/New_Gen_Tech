
# Perceptron, Loss Functions, and Connections to Logistic Regression

## 1. Perceptron Loss in `sklearn`
In `sklearn.linear_model.Perceptron`, the algorithm uses the **Perceptron loss**:

\[
L(w) = \max(0, -y (w \cdot x + b))
\]

where:
- \( y \in \{-1, +1\} \)
- \( w \cdot x + b = z \) (dot product with bias)

### Behavior:
- If \( y z > 0 \) → **correct prediction** → loss = 0  
- If \( y z < 0 \) → **wrong prediction** → loss increases linearly with misclassification margin.

Thus, the **distance of a point from the decision boundary** affects the loss:
- Correct & far away (large margin) → zero loss.
- Misclassified & far away → larger loss.

---

## 2. Gradient Descent on Perceptron Loss
We minimize loss by updating weights via **stochastic gradient descent**.

For misclassified points (\( y z < 0 \)):

\[
\frac{\partial L}{\partial w} = -y x, \quad \frac{\partial L}{\partial b} = -y
\]

Update rule:
\[
w \leftarrow w + \eta y x, \quad b \leftarrow b + \eta y
\]

This matches the update rule in your function.

```python
def perceptron(X,y):
    w1=w2=b=1
    lr = 0.1
    for j in range(1000):
        for i in range(X.shape[0]):
            z = w1*X[i][0] + w2*X[i][1] + b
            if z*y[i] < 0:  # misclassified
                w1 = w1 + lr*y[i]*X[i][0]
                w2 = w2 + lr*y[i]*X[i][1]
                b = b + lr*y[i]
    return w1,w2,b
```

---

## 3. Other Loss Functions in Classification
The location of a loss function in ML pipeline:
- **Model**: Computes raw score (\( z = w \cdot x + b \))
- **Activation function**: Converts raw score into meaningful output (probability or class)
- **Loss function**: Measures error between prediction and actual label
- **Optimizer (gradient descent)**: Updates weights to minimize loss

### Common Loss–Activation Pairs:
1. **Hinge Loss + Step Activation**
   - Used in perceptron & SVM.
   - Binary classification with hard margin.
   - Loss: \( L = \max(0, 1 - y z) \).

2. **Log Loss (Binary Cross Entropy) + Sigmoid Activation**
   - Used in Logistic Regression.
   - Outputs probability \( \hat{y} = \sigma(z) = \frac{1}{1+e^{-z}} \).
   - Loss:  
     \[ L = -y \log(\hat{y}) - (1-y) \log(1-\hat{y}) \]

3. **Categorical Cross Entropy + Softmax Activation**
   - Used in multi-class classification (Softmax Regression, Neural Nets).
   - Loss:  
     \[ L = -\sum_i y_i \log(\hat{y}_i) \]

4. **Mean Squared Error (MSE) + Linear Activation**
   - Used in regression tasks.
   - Loss:  
     \[ L = \frac{1}{N} \sum (y - \hat{y})^2 \]

---

## 4. Perceptron vs Logistic Regression
- **Perceptron**:
  - Activation: Step function (hard 0/1).
  - Loss: Perceptron/Hinge loss.
  - Only checks correct vs incorrect.

- **Logistic Regression**:
  - Activation: **Sigmoid** (smooth probability).
  - Loss: **Binary Cross Entropy (Log Loss)**.
  - Measures "how confident" prediction is.

When you change perceptron’s **activation** to sigmoid and **loss** to log loss, it becomes **Logistic Regression**.

---

## 5. Summary Table

| Model / Task                | Activation Function  | Output Type       | Loss Function                         |
|-----------------------------|----------------------|------------------|---------------------------------------|
| **Perceptron (Binary)**     | Step                 | Class (0/1)      | Perceptron Loss / Hinge Loss          |
| **Logistic Regression**     | Sigmoid              | Probability (0-1)| Binary Cross Entropy (Log Loss)       |
| **Softmax Regression**      | Softmax              | Probabilities     | Categorical Cross Entropy             |
| **Linear Regression**       | Linear               | Real value       | Mean Squared Error (MSE)              |

---

## 6. Key Insights
- Loss function defines **what is being optimized**.  
- Activation defines **how raw scores are mapped** to outputs.  
- Together they shape the **learning process** of ML models.
