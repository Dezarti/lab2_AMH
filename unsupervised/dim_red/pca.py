import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) implementation.

    Parameters:
    - n_components: int, Number of principal components to compute.

    Attributes:
    - components: array, Principal components extracted from the fit method.
    - mean: array, Mean of each feature in the dataset.
    """
    
    def __init__(self, n_components):
        """
        Initializes the PCA instance with the specified number of components.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """
        Fits the model to the data X by computing the principal components.

        Parameters:
        - X: array-like, shape (n_samples, n_features), Training data.
        """
        # Center the data by subtracting the mean of each feature
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute the covariance matrix of the centered data
        cov = np.cov(X_centered, rowvar=False)

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort the eigenvalues and corresponding eigenvectors in decreasing order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select the first n_components eigenvectors as the principal components
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        """
        Transforms the data X into the principal component space.

        Parameters:
        - X: array-like, shape (n_samples, n_features), Data to transform.

        Returns:
        - X_transformed: array, shape (n_samples, n_components), Data projected onto the principal components.
        """
        # Center the data by subtracting the mean
        X_centered = X - self.mean

        # Project the centered data onto the principal components
        X_transformed = np.dot(X_centered, self.components)

        return X_transformed
