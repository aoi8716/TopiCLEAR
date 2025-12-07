from .features import build_sentence_embeddings

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.decomposition import PCA
import numpy as np
from scipy.linalg import eigh
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from typing import Sequence, Optional, Any

try:
    from clustpy.alternative.nrkmeans import _update_centers_and_scatter_matrix as _clustpy_update
except ImportError:
    _clustpy_update = None

def _within_scatter_raw(X: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Compute the within scatter matrix for the given data and labels.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    labels : np.ndarray
        the cluster labels
    n_clusters : int
        the number of clusters

    Returns
    -------
    scatter : np.ndarray
        The within scatter matrix
    """
    n_features = X.shape[1]
    scatter = np.zeros((n_features, n_features))
    for k in range(n_clusters):
        cluster_points = X[labels == k]
        if cluster_points.size == 0:
            continue
        cluster_mean = np.mean(cluster_points, axis=0)
        centered_points = cluster_points - cluster_mean
        scatter += np.matmul(centered_points.T, centered_points)
    return scatter


def _topiclear(X: np.ndarray, n_clusters: int, n_dims: int, max_iter: int, gmm_repetitions: int,
            random_state: np.random.RandomState) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, GaussianMixture]:
    """
    Start the TopiCLEAR clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    n_clusters : int
        the number of clusters
    n_dims : int
        The number of features in the resulting subspace
    max_iter : int
        the maximum number of iterations
    gmm_repetitions : int
        Number of repetitions when executing GMM. For more information see sklearn.mixture.GaussianMixture (default: 10)
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution. Can also be of type int

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray, float, int, GaussianMixture)
        The labels as identified by TopiCLEAR,
        The final rotation matrix,
        The cluster centers in the subspace,
        The final error (using average negative log-likelihood from GMM),
        The number of iterations used for clustering,
        The fitted GaussianMixture object
    """
    assert n_clusters > 1, "n_clusters must be larger than 1"
    assert max_iter > 0, "max_iter must be larger than 0"
    if n_dims >= X.shape[1]:
        gmm = GaussianMixture(n_components=n_clusters, n_init=gmm_repetitions, random_state=random_state)
        gmm.fit(X)
        labels = gmm.predict(X)
        rotation = np.identity(X.shape[1])
        cluster_centers = gmm.means_
        error = -gmm.score(X)
        return labels, rotation, cluster_centers, error, 1, gmm

    # Global parameters
    global_mean = np.mean(X, axis=0)
    centered_points = X - global_mean
    St = np.matmul(centered_points.T, centered_points) / (X.shape[0] - 1)
    
    # Get initial rotation
    pca = PCA(n_dims)
    pca.fit(X)
    rotation = pca.components_.T
    
    #initial iteration
    X_subspace = np.matmul(X, rotation)
    gmm = GaussianMixture(n_components=n_clusters, n_init=gmm_repetitions, random_state=random_state)
    gmm.fit(X_subspace)
    old_labels = gmm.predict(X_subspace)
        
    
    # Repeat actions until convergence or max_iter
    for iteration in range(max_iter):
        # Update subspace
        if _clustpy_update is not None:
            _, scatter = _clustpy_update(X, n_clusters, old_labels)
            Sw = scatter / (X.shape[0] - 1)
        else:
            Sw = _within_scatter_raw(X, old_labels, n_clusters) / (X.shape[0] - 1)
            
        Sb = St - Sw
        try:
            _, eigen_vectors = eigh(Sb, Sw)
            # Take the eigenvectors with largest eigenvalues
            rotation = eigen_vectors[:, ::-1][:, :n_dims]
        except:
            # In case errors occur during eigenvalue decomposition keep algorithm running
            pass
        
        # Update labels
        X_subspace = np.matmul(X, rotation)
        gmm = GaussianMixture(n_components=n_clusters, n_init=gmm_repetitions, random_state=random_state)
        gmm.fit(X_subspace)
        new_labels = gmm.predict(X_subspace)
        # Check if labels have not changed
        if old_labels is not None and nmi(new_labels, old_labels) == 1:
            break
        else:
            old_labels = new_labels.copy()
            
    cluster_centers = gmm.means_
    error = -gmm.score(X_subspace)
    return new_labels, rotation, cluster_centers, error, iteration + 1, gmm


class TopiCLEAR(TransformerMixin, ClusterMixin, BaseEstimator):
    """
    Execute the TopiCLEAR clustering procedure.
    The initial rotation is normally the (n_clusters-1) components of a PCA.
    Afterward, GMM and LDA are executed one after the other until the labels do not change anymore.
    GMM always takes place in the rotated subspace.

    Parameters
    ----------
    n_clusters : int
        the number of clusters
    n_dims : int
        The number of features in the resulting subspace. If None this will be equal to n_clusters - 1 (default: None)
    max_iter : int
        the maximum number of iterations (default: 50)
    n_init : int
        number of times TopiCLEAR is executed using different seeds. The final result will be the one with lowest costs (default: 1)
    dim_preprocess : int
        PCA dimensionality used in the preprocessing stage before normalization (default: 64).
    random_state_pca : int | None
        Random seed passed to the preprocessing PCA (default: 50).
    gmm_repetitions : int
        Number of repetitions when executing GMM. For more information see sklearn.mixture.GaussianMixture (default: 10)
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)
    input_type : str
        The type of input data provided. Can be either "embedding" for pre-computed embeddings or "text" for raw text input. (default: "text")
        If "embedding", the input data must be a 2D numpy array of shape (n_samples, n_features).
    embedding_model_name : str
        The name of the pre-trained SentenceTransformer model to use when input_type is "text". (default: "sentence-transformers/all-MiniLM-L6-v2")
    embedding_model : Optional[Any]
        An optional pre-initialized SentenceTransformer model to use when input_type is "text". If provided, this model will be used instead of loading a new one. (default: None)
    embedding_batch_size : int
        The batch size for processing texts when input_type is "text". (default: 64)
    embedding_device : Optional[str]
        The device to run the embedding model on (e.g., "cpu" or "cuda") when input_type is "text". (default: None)
    embedding_show_progress_bar : bool
        Whether to display a progress bar during embedding when input_type is "text". (default: False)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels
    rotation_ : np.ndarray
        The final rotation matrix
    cluster_centers_ : np.ndarray
        The cluster centers in the rotated subspace
    error_ : float
        The final error (using average negative log-likelihood from GMM)
    n_features_in_ : int
        The number of features used for fitting

    References
    -------
    Ding, Chris, and Tao Li. "Adaptive dimension reduction using discriminant analysis and k-means clustering."
    Proceedings of the 24th international conference on Machine learning. 2007.
    """

    def __init__(self, n_clusters: int, n_dims: int = None, max_iter: int = 50, n_init: int = 1,
                gmm_repetitions: int = 10, random_state: np.random.RandomState | int = None, dim_preprocess: int = 64, random_state_pca: int | None = 50,
                input_type: str = "text", embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", embedding_model: Any | None = None, embedding_batch_size: int = 64, embedding_device: str | None = None, embedding_show_progress_bar: bool = False):
        self.n_clusters = n_clusters
        self.n_dims = n_clusters - 1 if n_dims is None else n_dims
        self.max_iter = max_iter
        self.n_init = n_init
        self.gmm_repetitions = gmm_repetitions
        self.random_state = check_random_state(random_state)
        self.dim_preprocess = dim_preprocess
        self.random_state_pca = random_state_pca
        
        if input_type not in {"text", "embedding"}:
            raise ValueError("input_type must be 'text' or 'embedding'")
        self.input_type = input_type
        self.embedding_model_name = embedding_model_name
        self.embedding_model = embedding_model
        self.embedding_batch_size = embedding_batch_size
        self.embedding_device = embedding_device
        self.embedding_show_progress_bar = embedding_show_progress_bar
    
    def _ensure_array(self, X: Sequence[Any] | np.ndarray) -> np.ndarray:
        if isinstance(X, np.ndarray):
            if X.size == 0:
                raise ValueError("Input array is empty.")
            if self.input_type != "embedding":
                raise ValueError("When providing a numpy array as input, input_type must be set to 'embedding'.")
            if X.ndim != 2:
                raise ValueError("Input array must be 2-dimensional of shape (n_samples, n_features).")
            return X
        
        if isinstance(X, str):
            raise ValueError("Input must be a sequence of strings or a numpy array, but a single string was provided.")
        
        if isinstance(X, Sequence) and len(X) > 0:
            if isinstance(X[0], str):
                if self.input_type == "text":
                    if self.embedding_model is None:
                        try:
                            from sentence_transformers import SentenceTransformer
                        except ImportError as e:
                            raise ImportError("Please install the 'sentence-transformers' package to use text embeddings.") from e
                        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.embedding_device)
                    vectors = build_sentence_embeddings(
                        texts=X,
                        model_name=self.embedding_model_name,
                        model=self.embedding_model,
                        batch_size=self.embedding_batch_size,
                        show_progress_bar=self.embedding_show_progress_bar,
                        device=self.embedding_device
                    )
                    return vectors
                else:
                    raise ValueError(f"Invalid input_type '{self.input_type}' for string inputs. Supported type is 'text'.")
            elif self.input_type == "embedding":
                arr = np.array(X, dtype=float)
                if arr.size == 0:
                    raise ValueError("Input sequence is empty.")
                if arr.ndim != 2:
                    raise ValueError("Input sequence must be convertible to a 2-dimensional numpy array of shape (n_samples, n_features).")
                return arr
            else:
                raise ValueError(f"Invalid input_type '{self.input_type}'. Supported types are 'embedding' and 'text'.")
        else:
            raise ValueError(
                "Input must be either:\n"
                "- a non-empty numpy array of shape (n_samples, n_features), or\n"
                "- a non-empty sequence of numeric vectors when input_type='embedding', or\n"
                "- a non-empty sequence of text strings when input_type='text'."
            )            

    def _fit_preprocessor(self, X: np.ndarray) -> None:
        n_samples, n_features = X.shape
        max_rank = max(1, min(n_features, n_samples - 1))
        n_components = min(self.dim_preprocess, max_rank)
        self.preprocess_pca_ = PCA(n_components=n_components, random_state=self.random_state_pca).fit(X)
        

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["preprocess_pca_"])
        reduced_vectors = self.preprocess_pca_.transform(X)
        norm_coef = np.sqrt(np.sum(reduced_vectors * reduced_vectors, axis=1).reshape(-1, 1))
        norm_coef[norm_coef == 0] = 1.0 # Avoid division by zero
        normalized_vectors = reduced_vectors / norm_coef        
        return normalized_vectors
    
    def _fit_on_array(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "TopiCLEAR":
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels are contained in the labels_ attribute.
        
        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can be ignored)
            
        Returns
        -------
        self : TopiCLEAR
            this instance of the TopiCLEAR algorithm
        """
        self._fit_preprocessor(X)
        normalized_vectors = self._preprocess(X)
        
        all_random_states = self.random_state.choice(10000, self.n_init, replace=False)
        best_costs = np.inf
        
        for i in range(self.n_init):
            local_random_state = check_random_state(all_random_states[i])
            labels, rotation, cluster_centers, error, n_iter, gmm = _topiclear(normalized_vectors, self.n_clusters, self.n_dims, self.max_iter,
                                                        self.gmm_repetitions,
                                                        local_random_state)
            if error < best_costs:
                best_costs = error
                # Update class variables
                self.labels_ = labels
                self.rotation_ = rotation
                self.cluster_centers_ = cluster_centers
                self.error_ = error
                self.n_features_in_ = X.shape[1] 
                self.n_iter_ = n_iter
                self.gmm_ = gmm
        
        return self
    
    def _transform_on_array(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input dataset with the rotation matrix identified by the fit function.
        
        Parameters
        ----------
        X : np.ndarray
            the given data set
            
        Returns
        -------
        rotated_data : np.ndarray
            The rotated data set
        """
        check_is_fitted(self, ["rotation_", "preprocess_pca_"])
        rotated_data = np.matmul(self._preprocess(X), self.rotation_)
        return rotated_data

    def fit(self, X: Sequence[str] | np.ndarray, y: Optional[np.ndarray] = None) -> "TopiCLEAR":
        """
        Initiate the actual clustering process on the input data set.

        Parameters
        ----------
        X : np.ndarray | Sequence[str]
            the given data set 
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        self : TopiCLEAR
            this instance of the TopiCLEAR algorithm
        """

        X_arr = self._ensure_array(X)
        self._fit_on_array(X_arr, y)                
        return self

    def transform(self, X: Sequence[str] | np.ndarray) -> np.ndarray:
        """
        Transform the input dataset with the rotation matrix identified by the fit function.

        Parameters
        ----------
        X : np.ndarray | Sequence[str]
            the given data set

        Returns
        -------
        rotated_data : np.ndarray
            The rotated data set
        """
        X_arr = self._ensure_array(X)
        return self._transform_on_array(X_arr)
        

    def predict(self, X: Sequence[str] | np.ndarray) -> np.ndarray:
        """
        Predict the labels of an input dataset.

        Parameters
        ----------
        X : np.ndarray | Sequence[str]
            the given data set

        Returns
        -------
        predicted_labels : np.ndarray
            the predicted labels of the input data set
        """
        X_arr = self._ensure_array(X)
        check_is_fitted(self, ["gmm_"])
        X_transformed = self._transform_on_array(X_arr)
        predicted_labels = self.gmm_.predict(X_transformed)
        return predicted_labels
    
    def predict_proba(self, X: Sequence[str] | np.ndarray) -> np.ndarray:
        """
        Predict the labels probabilities of an input dataset.

        Parameters
        ----------
        X : np.ndarray | Sequence[str]
            the given data set

        Returns
        -------
        predicted_label_probabilities : np.ndarray
            the predicted label probabilities of the input data set
        """
        X_arr = self._ensure_array(X)
        check_is_fitted(self, ["gmm_"])
        X_transformed = self._transform_on_array(X_arr)
        predicted_label_probabilities = self.gmm_.predict_proba(X_transformed)
        return predicted_label_probabilities

    def fit_transform(self, X: Sequence[str] | np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Fit the TopiCLEAR model and transform the input dataset.

        Parameters
        ----------
        X : np.ndarray | Sequence[str]
            the given data set
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        rotated_data : np.ndarray
            The rotated data set
        """
        X_arr = self._ensure_array(X)
        self._fit_on_array(X_arr, y)
        rotated_data = self._transform_on_array(X_arr)
        return rotated_data
    
    def score(self, X: Sequence[str] | np.ndarray) -> float:
        """
        Compute the negative log-likelihood of the input data under the fitted GMM.
        
        Parameters
        ----------
        X : np.ndarray | Sequence[str]
            the given data set
            
        Returns
        -------
        log_likelihood : float
            The negative log-likelihood of the input data
        """
        X_arr = self._ensure_array(X)
        check_is_fitted(self, ["gmm_"])
        X_transformed = self._transform_on_array(X_arr)
        log_likelihood = -self.gmm_.score(X_transformed)
        return float(log_likelihood)