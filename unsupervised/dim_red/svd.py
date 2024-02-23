import numpy as np

class SVD:
    """
    Singular Value Decomposition (SVD) implementation.

    Parameters:
    - n_components: int, Number of singular values and vectors to retain.

    Attributes:
    - U: array, Left singular vectors of the input data X.
    - S: array, Singular values of the input data X.
    - VT: array, Transpose of the right singular vectors of the input data X.
    """
    
    def __init__(self, n_components):
        """
        Initializes the SVD instance with the specified number of components.
        """
        self.n_components = n_components
        self.U = None
        self.S = None
        self.VT = None

    def fit(self, X):
        """
        Fits the SVD model to the data X by computing its singular value decomposition.

        Parameters:
        - X: array-like, shape (n_samples, n_features), The data to decompose.

        Returns:
        - self: object, Returns the instance itself.
        """
        # Compute SVD of the input data
        U, sigma, VT = np.linalg.svd(X, full_matrices=False)
        
        # Retain only the specified number of singular values and vectors
        self.U = U[:, :self.n_components]
        self.S = sigma[:self.n_components]
        self.VT = VT[:self.n_components, :]

        return self

    def transform(self, X):
        """
        Projects the data X onto the retained singular vectors.

        Parameters:
        - X: array-like, shape (n_samples, n_features), The data to project.

        Returns:
        - X_projected: array, shape (n_samples, n_components), The projected data.
        """
        # Project the input data onto the retained right singular vectors
        X_projected = np.dot(X, self.VT.T)
        
        return X_projected

    def fit_transform(self, X):
        """
        Fits the SVD model to the data X and then projects it onto the retained singular vectors.

        Parameters:
        - X: array-like, shape (n_samples, n_features), The data to fit and transform.

        Returns:
        - X_transformed: array, shape (n_samples, n_components), The transformed data.
        """
        self.fit(X)
        return self.transform(X)
