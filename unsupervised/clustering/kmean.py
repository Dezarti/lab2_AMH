import numpy as np

class KMeans:
    """
    KMeans clustering algorithm.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    max_iters : int
        Maximum number of iterations of the KMeans algorithm for a single run.

    Attributes
    ----------
    centroids : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : ndarray of shape (n_samples,)
        Labels of each point
    """
    
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels_ = None

    def initialize_centroids(self, data):
        """Randomly selects `n_clusters` data points as initial centroids."""
        indices = np.random.choice(data.shape[0], self.n_clusters, replace=False)
        self.centroids = data[indices]

    def assign_clusters(self, data):
        """Assigns each data point to the nearest centroid."""
        distances = np.sqrt(((data - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def update_centroids(self, data, assignments):
        """Updates centroids to be the mean of points assigned to each cluster."""
        new_centroids = np.array([data[assignments == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids

    def fit(self, data):
        """
        Computes KMeans clustering.

        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features)
            Training instances to cluster.
        """
        self.initialize_centroids(data)
        for _ in range(self.max_iters):
            assignments = self.assign_clusters(data)
            new_centroids = self.update_centroids(data, assignments)
            # Check for convergence
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        self.labels_ = assignments
        return self

    def predict(self, data):
        """
        Predicts the closest cluster each sample in `data` belongs to.

        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if self.centroids is None:
            raise ValueError("Model not yet fitted.")
        return self.assign_clusters(data)

# Example usage:
# data = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]])
# kmeans = KMeans(n_clusters=3, max_iters=100)
# kmeans.fit(data)
# print("Centroids:\n", kmeans.centroids)
# print("Assignments:\n", kmeans.labels_)
