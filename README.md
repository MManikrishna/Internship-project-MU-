# 🧠 Machine Learning Projects – Multiple Datasets

This repository contains implementations of various **Machine Learning algorithms** applied to multiple datasets during internship and project work at **Mahindra University**.  
The goal was to explore **supervised** and **unsupervised** learning techniques and apply them to real-world problems, including **Fashion Trend Analysis**.

---

## 📂 Datasets Used
- 🌸 **Iris Dataset** – Flower classification  
- 🚗 **Car Evaluation Dataset** – Vehicle acceptability classification  
- 🌰 **Dry Bean Dataset** – Bean type identification  
- 🛍️ **Mall Customer Dataset** – Customer segmentation  
- 🍄 **Mushroom Dataset** – Edible vs poisonous classification  
- 🍷 **Wine Quality Dataset** – Quality prediction  
- 🌱 **Plant Communication Dataset** – Plant trait analysis  
- 🧬 **Cancer Dataset (Denmark)** – Cancer type classification  
- 🧪 **Glass Classification Dataset** – Glass type prediction  
- 👗 **Fashion-MNIST Dataset** – Fashion trend analysis  

---

## ⚙️ Algorithms Implemented
### 🔹 Supervised Learning
- Decision Trees  
- Random Forests  
- Logistic Regression  
- Support Vector Machines (SVM)  
- k-Nearest Neighbors (k-NN)  
- Naïve Bayes  
- Neural Networks  

### 🔹 Unsupervised Learning
- K-Means Clustering  
- Hierarchical Clustering  
- DBSCAN  

---

## 🔄 Workflow
1. **Data Collection** – Import datasets from CSV/UCI/Kaggle.  
2. **Data Preprocessing** – Cleaning, normalization, encoding, handling missing values.  
3. **Exploratory Data Analysis (EDA)** – Visualization, distribution checks, correlations.  
4. **Modeling** – Apply supervised & unsupervised ML models.  
5. **Evaluation** –  
   - Classification: Accuracy, Precision, Recall, F1-score.  
   - Clustering: Silhouette Score.  
6. **Insights** – Understanding patterns, classifications, and clusters for decision-making.  

---

## 📊 Key Results
- **Random Forest** → Best performance in classification tasks (Accuracy ~90%+).  
- **Decision Tree** → Good but less effective with overlapping classes.  
- **K-Means** → Formed distinct clusters (Silhouette Score ~0.65).  
- **DBSCAN** → Useful for detecting rare/niche items.  

---

## 🚀 Future Improvements
- Incorporate **brand, price, seasonal features** for fashion datasets.  
- Apply **deep learning (CNNs)** for image-based fashion analysis.  
- Develop **interactive dashboards** for visualization.  
- Use **feature selection techniques** to improve accuracy.  

---

## 🛠️ Tools & Technologies
- **Python** (Jupyter Notebook)  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras  

---

## 🙏 Acknowledgements
- **Mahindra University** – Internship & guidance  
- **Faculty Mentors** – Prof. Dr. Arun K. Pujari, Dr. Tauheed Ahmed, Dr. Shabnam Samima  
- **Dr. Motahar Reza (GITAM)** – Provided opportunity  

---

## 📌 How to Run
1. Clone the repo:  
   ```bash
   git clone https://github.com/your-username/ml-multiple-datasets.git
   cd ml-multiple-datasets



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
