#read mnist.npz and perform PCA dimensionality reduction to dimension 2 and 3 and then plot the data in 2D and 3D respectively coloring the points apparteining to the same cluster (i.e. have the same label) with the same color.

# Path: PCA_Analysis.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import numpy as np

# read data from file
data = np.load('mnist.npz')
X = data['X'].astype(np.float32)
y = data['y'].astype(np.float32)
print("Data loaded")
#Perform PCA dimensionality reduction to dimension 1
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)

#Plot the data in 1D
fig, ax = plt.subplots(figsize=(10, 7))
for j in range(10):
    ax.scatter(X_pca[y == j, 0], np.zeros(X_pca[y == j, 0].shape), label='y = {}'.format(j))
ax.legend()
plt.suptitle('PCA n_components: 1')
plt.savefig('PCA1.png')


#Perform PCA dimensionality reduction to dimension 2 and 3
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

#Plot the data in 2D
fig, ax = plt.subplots(figsize=(10, 7))
for j in range(10):
    ax.scatter(X_pca[y == j, 0], X_pca[y == j, 1], label='y = {}'.format(j))
ax.legend()
plt.suptitle('PCA n_components: 2')
plt.savefig('PCA2.png')

#Perform PCA dimensionality reduction to dimension 3
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

#Plot the data in 3D with different rotation
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
for j in range(10):
    ax.scatter(X_pca[y == j, 0], X_pca[y == j, 1], X_pca[y == j, 2], label='y = {}'.format(j))
ax.legend()
plt.suptitle('PCA n_components: 3')
plt.savefig('PCA3.png')

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
for j in range(10):
    ax.scatter(X_pca[y == j, 0], X_pca[y == j, 1], X_pca[y == j, 2], label='y = {}'.format(j))
ax.view_init(30, 30)
ax.legend()
plt.suptitle('PCA n_components: 3')
plt.savefig('PCA3_2.png')

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
for j in range(10):
    ax.scatter(X_pca[y == j, 0], X_pca[y == j, 1], X_pca[y == j, 2], label='y = {}'.format(j))
ax.view_init(30, 60)
ax.legend()
plt.suptitle('PCA n_components: 3')
plt.savefig('PCA3_3.png')

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
for j in range(10):
    ax.scatter(X_pca[y == j, 0], X_pca[y == j, 1], X_pca[y == j, 2], label='y = {}'.format(j))
ax.view_init(30, 90)
ax.legend()
plt.suptitle('PCA n_components: 3')
plt.savefig('PCA3_4.png')



