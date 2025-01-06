import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Imports the data
data = pd.read_csv("part5-unsupervised-learning/customer_data.csv")
x = data[["Annual Income", "Spending Score"]]  # Select only numeric columns for clustering

# Standardize the data
x_std = StandardScaler().fit_transform(x)

# The value of k has been defined for you
k = 5

# Apply the kmeans algorithm
km = KMeans(n_clusters=k).fit(x_std)

# Get the centroid and label values
centroids = km.cluster_centers_
labels = km.labels_

# Sets the size of the graph
plt.figure(figsize=(5,4))

# Use a for loop to plot the data points in each cluster
for i in range(k):
    cluster = x_std[labels == i]
    plt.scatter(cluster[:, 0], cluster[:, 1])

# Plot the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=100, c='r', label='centroid')

# Shows the graph
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()