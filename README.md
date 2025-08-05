# 🌳 Decision Tree Learning Projects

Welcome to the repository for all my Decision Tree learning projects!  
This includes intuitive explanations, Python implementations, and story-based learning to understand how decision trees work, when they stop, and how they behave on real datasets.

---

## 📌 Overview

This repository focuses on:
- Building and visualizing Decision Trees using Python and scikit-learn.
- Learning how Decision Trees work through simple **5-step logic** and **stories**.
- Applying the model to datasets like **Iris**.
- Understanding key concepts like Gini index, entropy, overfitting, stopping conditions, and tree depth.

---

## 🧠 What is a Decision Tree?

A **Decision Tree** is a supervised machine learning model used for classification or regression.  
It works by **splitting data** based on feature values and making decisions at each step until a **final output (leaf node)** is reached.

Think of it like playing **20 Questions** to guess the right answer. 🎯

---

## ✨ Topics Covered

### ✅ 1. Decision Tree in 5 Steps
1. Select the best feature to split (based on Gini or Entropy)
2. Split the data into branches
3. Repeat for each subset
4. Stop when:
   - All samples are of the same class
   - No features left
   - Tree is deep enough
   - Not enough samples to split
   - No gain from further splits
5. Assign class labels at leaf nodes

---

### 📖 2. Story-Based Understanding

To make learning fun and memorable:
- **Tina the Tree** asks questions and stops when she's sure.
- **Captain Tree** follows smart stopping rules.
- **Detective Dot** classifies flowers based on petal and sepal clues.

---

### 📊 3. Dataset Used – Iris Dataset

- Features:
  - Petal Length & Width
  - Sepal Length & Width
- Classes:
  - Setosa
  - Versicolor
  - Virginica

---

### 🧪 4. Code Example (Iris Dataset)
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

plot_tree(clf, filled=True)
plt.show()
