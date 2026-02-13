# 📘 Naïve Bayes Classifier – Notes

## 1. Introduction
Naïve Bayes is a **probabilistic classifier** based on **Bayes’ Theorem**.  
It assumes **independence between features** (hence "Naïve").  
It is commonly used for **text classification, spam filtering, sentiment analysis, etc.**

---

## 2. Bayes Theorem
For two events **A** and **B**:

\[
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
\]

- **\( P(A|B) \): Posterior probability** → Probability of event A given event B.  
- **\( P(B|A) \): Likelihood** → Probability of event B given A.  
- **\( P(A) \): Prior probability** → Probability of event A.  
- **\( P(B) \): Marginal probability** → Probability of event B.  

---

## 3. Independent & Dependent Events
- **Independent Events:**  
  Two events A and B are independent if:  
  \[
  P(A \cap B) = P(A) \cdot P(B)
  \]

- **Dependent Events (Conditional Probability):**  
  When events are related:  
  \[
  P(A|B) = \frac{P(A \cap B)}{P(B)}
  \]

---

## 4. Derivation of Bayes Theorem
From conditional probability definition:  

\[
P(A|B) = \frac{P(A \cap B)}{P(B)} \quad \text{and} \quad P(B|A) = \frac{P(A \cap B)}{P(A)}
\]

Equating both:

\[
P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)
\]

Rearranging gives **Bayes Theorem**:

\[
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
\]

---

## 5. Naïve Bayes Classifier Mapping
We want to classify input features \((x_1, x_2, ..., x_n)\) into a class \(y\).

- Classes: \( y \in \{C_1, C_2, ..., C_k\} \)  
- Features: \( x_1, x_2, ..., x_n \)

We compute:

\[
P(y|x_1, x_2, ..., x_n) = \frac{P(x_1, x_2, ..., x_n | y) \cdot P(y)}{P(x_1, x_2, ..., x_n)}
\]

Using **Naïve assumption** (independence of features):

\[
P(y|x_1, x_2, ..., x_n) \propto P(y) \cdot \prod_{i=1}^n P(x_i|y)
\]

**Decision Rule:**
\[
y = \arg\max_{C_k} P(C_k) \cdot \prod_{i=1}^n P(x_i | C_k)
\]

---

## 6. Steps to Solve Naïve Bayes
1. **Prepare dataset:** Identify features \((x_1, x_2, ..., x_n)\) and output class \(y\).  
2. **Calculate Priors:** \( P(C_k) = \frac{\text{count of class } C_k}{\text{total samples}} \).  
3. **Calculate Likelihoods:** For each feature value given the class:  
   \[
   P(x_i|C_k) = \frac{\text{count}(x_i, C_k)}{\text{count}(C_k)}
   \]
4. **Apply Naïve Bayes Formula:**  
   \[
   P(C_k|x_1,...,x_n) \propto P(C_k) \cdot \prod_{i=1}^n P(x_i|C_k)
   \]
5. **Predict Output:** Choose class with maximum posterior probability.  

---

## 7. Example (Multiple Inputs, Multiple Outputs)
Suppose we classify whether an **email** is **Spam (S)** or **Not Spam (N)**.  

- Features:  
  - \(x_1 =\) contains word "Free"  
  - \(x_2 =\) contains word "Offer"  
  - \(x_3 =\) contains word "Click"  

We calculate:  

\[
P(S|x_1, x_2, x_3) \propto P(S) \cdot P(x_1|S) \cdot P(x_2|S) \cdot P(x_3|S)
\]

\[
P(N|x_1, x_2, x_3) \propto P(N) \cdot P(x_1|N) \cdot P(x_2|N) \cdot P(x_3|N)
\]

- Whichever probability is **higher** → predicted class.

---

## 8. Key Points
- **Linear time complexity**: Scales well with data.  
- **Assumption of independence** is often violated, but still works surprisingly well.  
- Works best with **categorical data** (but Gaussian Naïve Bayes handles continuous data).  
- Common types:  
  - **Multinomial Naïve Bayes** (text classification).  
  - **Bernoulli Naïve Bayes** (binary features).  
  - **Gaussian Naïve Bayes** (continuous features).  
