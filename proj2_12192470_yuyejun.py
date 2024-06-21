import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

ratings = np.loadtxt('ratings.dat', delimiter='::', dtype=int)

n_users = np.max(ratings[:, 0])
n_movies = np.max(ratings[:, 1])
user_item_matrix = np.zeros((n_users, n_movies))

for user_id, movie_id, rating, _ in ratings:
    user_item_matrix[user_id-1, movie_id-1] = rating

km = KMeans(n_clusters=3, random_state=0)
km.fit(X)
labels = km.labels_
y_km=km.predict(X)
y_km

plt.scatter(X[y_km==0, 0], X[y_km==0, 1],
            s=50, c='lightgreen', marker='s',edgecolor='black', label='cluster1'
)

plt.scatter(X[y_km==1, 0], X[y_km==1, 1],
            s=50, c='orange', marker='o',edgecolor='black', label='cluster2'
)

plt.scatter(X[y_km==2, 0], X[y_km==2, 1],
            s=50, c='lightblue', marker='v',edgecolor='black', label='cluster3'
)

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],
           s=250, marker="*", c='red', edgecolor='black', label='centroids'
)

plt.legend(scatterpoints=1)
plt.grid()
plt.show()