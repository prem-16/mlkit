import numpy as np


np.random.seed(42)

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:

    def __init__(self,K =5, max_iters=100, plot_steps =False) -> None:
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps


        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]

        #mean feature vector for each cluster
        self.centroids = []

    
    def predict(self, X):
        self.X = X
        self.n_samples ,self.n_features = X.shape

        # initialize centroids 
        random_sample_idxs = np.random.choice(self.n_samples, self.K , replace = False)
        self.centroids =[self.X[idx] for idx in random_sample_idxs]

        # optimization
        for _ in range(self.max_iters):
            # update clusters
            self.clusters = self._create_clusters(self.centroids)
            # update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            # check if converged
    
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
    
    def _closest_centroid(self, sample,centroids):
        distances = [euclidean_distance(sample,point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.k, self.n_features))
        # return cluster labels
            