from shapely.geometry import Polygon, LineString
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.cluster import DBSCAN

rect_width = 4

with open('points_out.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    points = [(float(row[0]), float(row[1])) for row in reader]

points = np.array(points)
db = DBSCAN(eps=0.2, min_samples=5).fit(points)

# Extract the labels (clusters)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print(f'Estimated number of clusters: {n_clusters_}')
print(f'Estimated number of noise points: {n_noise_}')

# Plot the clusters
plt.figure(figsize=(8, 6))
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = points[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

plt.title(f'Estimated number of clusters: {n_clusters_}')
plt.xlabel('x')
plt.ylabel('z')
plt.grid(True)
plt.show()