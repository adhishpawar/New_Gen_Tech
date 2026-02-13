# 📘 K-Nearest Neighbors (KNN) – Notes

## 1. Introduction
- **K-Nearest Neighbors (KNN)** is a **non-parametric, lazy learning algorithm**.  
- Used for both:
  - **Classification** → assigns class label based on majority vote of nearest neighbors.  
  - **Regression** → predicts value by averaging nearest neighbors.  

---

## 2. Distance Calculation
KNN depends on finding **nearest neighbors** using a distance metric.

1. **Euclidean Distance** (most common):  
   \[
   d(p, q) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}
   \]  
   → Straight-line distance between two points.

2. **Manhattan Distance** (city-block distance):  
   \[
   d(p, q) = \sum_{i=1}^n |p_i - q_i|
   \]  
   → Sum of absolute differences.

---

## 3. Hyperparameter: **K**
- **K** = number of neighbors considered.  
- Small \(K\) → more sensitive to noise/outliers.  
- Large \(K\) → smoother decision boundary, but may ignore local patterns.  
- **Selection of K:**
  1. Try different K values.  
  2. Compute error rate (or accuracy).  
  3. Choose K with **lowest error rate / highest accuracy**.  

---

## 4. Error Rate & Model Selection
- For classification:  
  \[
  \text{Error Rate} = \frac{\text{Number of misclassified samples}}{\text{Total samples}}
  \]  
- Plot error vs. K → pick optimal K.  
- Typically, odd K is chosen to avoid ties in binary classification.  

---

## 5. Limitations
- **Works poorly with:**
  - **Outliers** → can distort nearest neighbor calculation.  
  - **Imbalanced datasets** → dominant class outweighs minority class in majority voting.  
- Computationally expensive for large datasets (distance must be calculated for all points).  

---

## 6. Real-Life Applications
- **Recommendation Systems** (finding similar users/items).  
- **Medical Diagnosis** (classifying diseases based on symptoms).  
- **Finance** (predicting loan approval or default risk).  
- **Image Recognition** (classifying objects based on pixel similarity).  
- **Customer Segmentation** (grouping customers with similar behavior).  

---

✅ **Summary:**  
- KNN is simple, intuitive, and effective for small-to-medium datasets.  
- Choosing **K** and distance metric is critical.  
- Sensitive to outliers & imbalanced data, but widely used in practical ML applications.  
