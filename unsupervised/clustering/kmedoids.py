import numpy as np

class KMedoids:
    """
    K-Medoids clustering.
    
    Parameters:
    - n_clusters: int, The number of clusters to form as well as the number of medoids to generate.
    - max_iter: int, Maximum number of iterations of the K-Medoids algorithm for a single run.
    - random_state: int, Random state for reproducibility.
    
    Attributes:
    - medoids: array, [n_clusters, n_features] Medoids coordinates.
    - labels_: array, [n_samples] Index of the cluster each sample belongs to.
    """
    def __init__(self, n_clusters=3, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.medoids = None
        self.labels_ = None

    def fit(self, X):
        """
        Compute K-Medoids clustering.
        
        Parameters:
        - X: array-like or sparse matrix, shape = [n_samples, n_features]
        
        Returns:
        self
        """
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        # Randomly initialize medoids
        initial_medoids_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.medoids = X[initial_medoids_indices]

        for _ in range(self.max_iter):
            # Compute distances and assign labels
            distances = np.sqrt(((X[:, np.newaxis, :] - self.medoids[np.newaxis, :, :]) ** 2).sum(axis=2))
            self.labels_ = np.argmin(distances, axis=1)

            # Update medoids
            new_medoids = np.array([X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])
            
            # Check for convergence
            if np.all(new_medoids == self.medoids):
                break
            self.medoids = new_medoids

        return self

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        
        Parameters:
        - X: array-like or sparse matrix, shape = [n_samples, n_features]
        
        Returns:
        labels: array, shape = [n_samples]
        """
        if self.medoids is None:
            raise ValueError("Model not yet fitted.")
        distances = np.sqrt(((X[:, np.newaxis, :] - self.medoids[np.newaxis, :, :]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

