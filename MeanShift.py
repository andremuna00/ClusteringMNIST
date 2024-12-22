from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift
from sklearn.metrics import rand_score
from sklearn.cluster import estimate_bandwidth
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# read data from file
data = np.load('mnist.npz')
X = data['X'].astype(np.float32)
y = data['y'].astype(np.float32)
#take a subset of the data half of the data
X = X[:30000]
y = y[:30000]
print("Data loaded")

n_components_range = [2, 50, 100, 150, 200]
#various kernel bandwidths
bandwidth_range = [5, 10, 20, 100]

# find the best bandwidth with estimate_bandwith()
#best_bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=30000)
#print("Best bandwidth: ", best_bandwidth)
# Initialize arrays for rand scores and learning times
rand_scores = np.zeros((len(n_components_range), len(bandwidth_range)))
learning_times = np.zeros((len(n_components_range), len(bandwidth_range)))

# Loop over the PCA dimensionality reduction range
for i, n_components in enumerate(n_components_range):
    # Apply PCA dimensionality reduction
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Loop over the k values for clustering
    for j, k in enumerate(bandwidth_range):
            
            # Apply Spectral Clustering
            ms = MeanShift(bandwidth=k)
    
            # Compute the rand score and learning time
            t0 = time.time()
            y_pred = ms.fit_predict(X_pca)
            t1 = time.time()
            rand_score_computed = rand_score(y, y_pred)
            learning_time = t1 - t0
    
            # Store the rand score and learning time
            rand_scores[i, j] = rand_score_computed
            learning_times[i, j] = learning_time
    
            # Print the results
            print('PCA n_components: {}, k: {}, rand score: {}, learning time: {}s'.format(n_components, k, rand_score_computed, learning_time))

# Plot the rand scores as a function of PCA dimensionality and k value
fig, ax = plt.subplots(figsize=(10, 7))
for j, k in enumerate(bandwidth_range):
    ax.plot(n_components_range, rand_scores[:, j], label='k = {}'.format(k))
ax.set_xlabel('PCA n_components')
ax.set_ylabel('Rand score')
ax.set_title('Rand score as a function of PCA dimensionality and k kernel bandwidth')
ax.legend()
#save the plots
plt.savefig('MeanShiftRandScore.png')
plt.show()

# Plot the learning times as a function of PCA dimensionality and k value
fig, ax = plt.subplots(figsize=(10, 7))
for j, k in enumerate(bandwidth_range):
    ax.plot(n_components_range, learning_times[:, j], label='k = {}'.format(k))
ax.set_xlabel('PCA n_components')
ax.set_ylabel('Learning time (s)')
ax.set_title('Learning time as a function of PCA dimensionality and k kernel bandwidth')
ax.legend()
#save the plots
plt.savefig('MeanShiftPCA.png')
plt.show()

# Find the best rand score and the corresponding PCA dimensionality and k value
best_rand_score = np.max(rand_scores)
best_n_components = n_components_range[np.argmax(rand_scores) // len(bandwidth_range)]
best_k = bandwidth_range[np.argmax(rand_scores) % len(bandwidth_range)]
print('Best rand score: {}, PCA n_components: {}, k: {}'.format(best_rand_score, best_n_components, best_k))

#save all the data to a file
np.savez('MeanShift.npz', rand_scores=rand_scores, learning_times=learning_times, n_components_range=n_components_range, k_bandwidth=bandwidth_range)


