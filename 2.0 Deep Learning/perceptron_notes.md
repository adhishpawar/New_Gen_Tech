
# Perceptron Implementation – Explanation

## 1. Dataset Creation
- The dataset is generated using `make_classification` from `sklearn.datasets`.
- Parameters:
  - `n_samples=100`: Number of data points.
  - `n_features=2`: Two input features (2D space for easy visualization).
  - `n_classes=2`: Binary classification problem.
  - `class_sep=10`: Ensures the two classes are linearly separable.

This setup gives us a clean dataset for testing perceptron learning.

---

## 2. Perceptron Function
The perceptron algorithm is implemented manually in a function.

### Steps inside the function:
1. **Bias Inclusion**:  
   - A bias term is added by inserting a column of 1’s at the start of `X`.  
   - This allows the perceptron to shift the decision boundary up or down.

2. **Initialization of Weights**:  
   - All weights are initialized to `1`.  
   - Learning rate (`lr`) is set to `0.1`.

3. **Training Loop**:
   - For 1000 iterations, a random sample `j` is chosen.
   - Weighted sum:  
     \\( z = X[j] \cdot w \\)
   - Step activation function is applied:  
     \\( y\_hat = 1 \, \text{if} \, z > 0 \; \text{else} \, 0 \\)
   - Weight update rule:  
     \\( w = w + \eta (y_{true} - y_{pred})X[j] \\)

4. **Return Values**:
   - Returns **intercept** (bias weight) and **coefficients** (feature weights).

---

## 3. Decision Boundary Calculation
- After training, the line equation separating the two classes is derived from weights.

General equation:  
\[ w_0 + w_1 x_1 + w_2 x_2 = 0 \]

Rearranging for 2D:  
\[ x_2 = -\frac{w_1}{w_2}x_1 - \frac{w_0}{w_2} \]

Here:
- **Slope (m)** = \\( -\frac{w_1}{w_2} \\)
- **Intercept (b)** = \\( -\frac{w_0}{w_2} \\)

This line is plotted as the decision boundary.

---

## 4. Visualization
- Blue/green dots: input data points from two classes.
- Red line: decision boundary learned by the perceptron.

Interpretation:
- Points above the line → classified as Class 1.
- Points below the line → classified as Class 0.

---

## 5. Key Takeaways
- The perceptron updates weights iteratively to minimize misclassifications.
- The decision boundary depends on the **weights (orientation)** and **bias (position)**.
- Works well only if the data is **linearly separable**.
