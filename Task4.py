import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph

# Step 1: Create a synthetic dataset for classification
# We generate 1,000 samples with 10 features. The dataset contains 6 classes, 
# with each class forming one cluster. 5 out of 10 features are informative for classification.
X, y = make_classification(
    n_samples=1000, n_features=10, n_classes=6, n_clusters_per_class=1, n_informative=5, random_state=42
)

# Step 2: Split the dataset into training and testing subsets
# Use 70% of the data for training and 30% for testing. Random state ensures reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Standardize the features
# Standardization scales the data so that all features have a mean of 0 and a standard deviation of 1.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Perform K-Means clustering on the training data
# We divide the training data into 20 clusters using K-Means. Each cluster is identified with a unique label.
num_clusters = 20
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_train)

# Step 5: Calculate density of each cluster
# Cluster density is calculated using the distance between points and their neighbors.
# Higher density indicates closely packed data points.
def calculate_density(X_train, cluster_labels, num_neighbors=5):
    densities = []
    for cluster_id in np.unique(cluster_labels):
        # Extract points belonging to the current cluster
        cluster_points = X_train[cluster_labels == cluster_id]
        # Create a graph representing the distances between points
        graph = kneighbors_graph(cluster_points, num_neighbors, mode='distance')
        distances = graph.data
        # Calculate density as the inverse of the sum of distances
        density = 1 / np.sum(distances) if len(distances) > 0 else 0
        densities.append((cluster_id, density))
    # Sort clusters by density in descending order
    return sorted(densities, key=lambda x: x[1], reverse=True)

# Get densities for all clusters
densities = calculate_density(X_train, cluster_labels)

# Eliminate dense clusters
# Here, we remove the 10 densest clusters and retain the rest for single-stage sampling.
dense_clusters_to_remove = 10
remaining_clusters = [c[0] for c in densities[dense_clusters_to_remove:]]

# Create a new training set with only the selected clusters
X_train_ss = X_train[np.isin(cluster_labels, remaining_clusters)]
y_train_ss = y_train[np.isin(cluster_labels, remaining_clusters)]

# Step 6: Perform double-stage sampling
# Randomly select 50% of the remaining data points for further sampling.
np.random.seed(42)
sample_size = int(len(X_train_ss) * 0.5)
indices = np.random.choice(len(X_train_ss), size=sample_size, replace=False)
X_train_ds = X_train_ss[indices]
y_train_ds = y_train_ss[indices]

# Step 7: Define the MLP classifier
# This MLP has two hidden layers with 200 and 100 neurons respectively.
# It is configured to stop early if validation performance does not improve.
mlp = MLPClassifier(
    hidden_layer_sizes=(200, 100), max_iter=10000, solver='adam',
    learning_rate_init=0.01, alpha=0.00001, batch_size=64, random_state=42,
    tol=1e-4, early_stopping=True, validation_fraction=0.0001
)

# Train and evaluate the MLP on the original dataset
start_time = time.time()
mlp.fit(X_train, y_train)
train_time_orig = time.time() - start_time
accuracy_orig = accuracy_score(y_test, mlp.predict(X_test))

# Train and evaluate the MLP on single-stage sampled data
start_time = time.time()
mlp.fit(X_train_ss, y_train_ss)
train_time_ss = time.time() - start_time
accuracy_ss = accuracy_score(y_test, mlp.predict(X_test))

# Train and evaluate the MLP on double-stage sampled data
start_time = time.time()
mlp.fit(X_train_ds, y_train_ds)
train_time_ds = time.time() - start_time
accuracy_ds = accuracy_score(y_test, mlp.predict(X_test))

# Step 8: Visualize the results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot original training data
axes[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired)
axes[0, 0].set_title("Original Data")

# Plot K-Means clusters
axes[0, 1].scatter(X_train[:, 0], X_train[:, 1], c=cluster_labels, cmap=plt.cm.Paired)
axes[0, 1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X')
axes[0, 1].set_title("K-Means Clusters")

# Plot single-stage sampled data
axes[1, 0].scatter(X_train_ss[:, 0], X_train_ss[:, 1], c=y_train_ss, cmap=plt.cm.Paired)
axes[1, 0].set_title("Single-Stage Sampling")

# Plot double-stage sampled data
axes[1, 1].scatter(X_train_ds[:, 0], X_train_ds[:, 1], c=y_train_ds, cmap=plt.cm.Paired)
axes[1, 1].set_title("Double-Stage Sampling")

plt.tight_layout()
plt.show()

# Step 9: Print the evaluation results
# The results include accuracy and training time for models trained on three datasets.
print(f"(Original Data) Mean Testing Accuracy: {accuracy_orig:.3f}, Training Time: {train_time_orig*1000:.3f} ms")
print(f"(Single-Stage Clustering) Mean Testing Accuracy: {accuracy_ss:.3f}, Training Time: {train_time_ss*1000:.3f} ms")
print(f"(Double-Stage Clustering) Mean Testing Accuracy: {accuracy_ds:.3f}, Training Time: {train_time_ds*1000:.3f} ms")
