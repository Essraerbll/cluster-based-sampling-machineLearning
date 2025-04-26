# Cluster Based Sampling for Efficient Neural Network Training

---

## 📚 Project Overview

This project addresses the challenge of reducing training time in machine learning classification tasks without significantly compromising accuracy.  
By leveraging **K-Means clustering** and **cluster density analysis**, the project applies **single-stage** and **double-stage sampling strategies** to selectively reduce the size of the training dataset.

A **Multi-Layer Perceptron (MLPClassifier)** is then trained and evaluated on the original and sampled datasets to measure the trade-offs between model performance and computational efficiency.

---

## 🛠 Methods and Tools

- **Synthetic Data Generation:**  
  - Classification dataset created using `make_classification` (1,000 samples, 10 features, 6 classes).

- **Preprocessing:**  
  - Standardization of features using **StandardScaler**.

- **Clustering:**  
  - **K-Means clustering** (`n_clusters=20`) applied on the training data.
  - Cluster **densities** calculated using **k-nearest neighbors graph** distances.

- **Sampling Strategies:**  
  - **Single-Stage Sampling:** Removing the densest clusters to retain more diverse data.
  - **Double-Stage Sampling:** Additional random sampling applied to the reduced dataset.

- **Model Training:**  
  - **MLPClassifier** with two hidden layers (200 and 100 neurons).
  - Early stopping enabled to prevent overfitting.

- **Evaluation Metrics:**  
  - Testing accuracy
  - Training time (measured in milliseconds)

- **Visualization:**  
  - Scatter plots illustrating data distributions at each sampling stage.

---

## 📦 File Structure

| File Name   | Purpose                                               |
|-------------|-------------------------------------------------------|
| `Task4.py`  | Main script for clustering, sampling, training, and evaluation |

No external data files are required; the dataset is synthetically generated within the script.

---

## 🚀 How to Run

1. Install required packages:
```bash
pip install numpy pandas scikit-learn matplotlib
```

2. Execute the script:
```bash
python Task4.py
```

Outputs include:
- Visualizations of original, clustered, single-stage, and double-stage sampled datasets.
- Printed metrics: training time and testing accuracy for each data subset.

---

## 📈 Key Findings

- **Cluster density analysis** effectively identifies redundant, densely packed data points.
- **Single-stage sampling** removes redundant clusters, leading to reduced training time with minimal accuracy loss.
- **Double-stage sampling** further reduces data volume but may slightly decrease accuracy.
- **Training time** was substantially reduced without significant performance degradation.

---

## ✨ Motivation

In many real-world scenarios, training on the entire dataset is computationally expensive and inefficient.  
By intelligently selecting representative samples based on cluster properties, it is possible to:
- **Accelerate training**
- **Reduce computational resources**
- **Maintain acceptable model performance**

This project showcases the potential of **density-driven sampling techniques** to improve machine learning pipelines.

---

## 🧠 Future Work

- Apply the method to real-world large-scale datasets (e.g., CIFAR-10, ImageNet subsets).
- Explore other clustering algorithms (e.g., DBSCAN, Gaussian Mixture Models).
- Combine density-based sampling with **active learning** strategies for dynamic model updating.

---

## 📢 Acknowledgements

This project is inspired by concepts from **data reduction**, **efficient training**, and **representation learning** fields within machine learning research.

---

# 🔥 Academic Keywords

> Data Sampling, Cluster Density, K-Means Clustering, MLPClassifier, Training Efficiency, Single-Stage Sampling, Double-Stage Sampling, Neural Network Training Optimization, Synthetic Dataset

---
