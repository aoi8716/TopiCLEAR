# The LDA-KMeans procedure in this module is adapted from ClustPy 0.0.2.
# ClustPy is distributed under the BSD-3-Clause License; see
# THIRD_PARTY_LICENSES/ClustPy-LICENSE.

from .topiclear import TopiCLEAR, _clustpy_update, _within_scatter_raw

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as nmi
import numpy as np
from scipy.linalg import eigh
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from typing import Sequence, Optional, Any


def _topiclear_kmeans(
        X: np.ndarray,
        n_clusters: int,
        n_dims: int,
        max_iter: int,
        kmeans_repetitions: int,
        random_state: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, KMeans, np.ndarray]:
    """
    Start the K-means variant of the TopiCLEAR clustering procedure.

    Returns
    -------
    tuple
        The final labels,
        the final LDA rotation,
        the final KMeans cluster centers,
        the final KMeans inertia,
        the final fitted KMeans object,
        and the rotation used to fit that final KMeans object.
    """
    assert n_clusters > 1, "n_clusters must be larger than 1"
    assert max_iter > 0, "max_iter must be larger than 0"

    if n_dims >= X.shape[1]:
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=kmeans_repetitions,
            random_state=random_state
        )
        kmeans.fit(X)

        rotation = np.identity(X.shape[1])
        prediction_rotation = rotation.copy()

        return (
            kmeans.labels_,
            rotation,
            kmeans.cluster_centers_,
            kmeans.inertia_,
            kmeans,
            prediction_rotation
        )

    old_labels = None

    global_mean = np.mean(X, axis=0)
    centered_points = X - global_mean
    St = centered_points.T @ centered_points / (X.shape[0] - 1)

    pca = PCA(n_dims)
    pca.fit(X)
    rotation = pca.components_.T

    for iteration in range(max_iter):
        prediction_rotation = rotation

        X_subspace = X @ rotation

        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=kmeans_repetitions,
            random_state=random_state
        )
        kmeans.fit(X_subspace)

        new_labels = kmeans.labels_

        if (
            old_labels is not None
            and nmi(new_labels, old_labels) == 1
        ):
            break

        old_labels = new_labels.copy()

        if _clustpy_update is not None:
            _, scatter = _clustpy_update(
                X,
                n_clusters,
                new_labels
            )
            Sw = scatter / (X.shape[0] - 1)
        else:
            Sw = _within_scatter_raw(
                X,
                new_labels,
                n_clusters
            ) / (X.shape[0] - 1)

        Sb = St - Sw

        try:
            _, eigen_vectors = eigh(Sb, Sw)
            rotation = eigen_vectors[
                :, ::-1
            ][:, :n_dims]
        except:
            pass

    return (
        new_labels,
        rotation,
        kmeans.cluster_centers_,
        kmeans.inertia_,
        kmeans,
        prediction_rotation
    )


class TopiCLEARKMeans(TopiCLEAR):
    """
    Execute the K-means variant of the TopiCLEAR clustering procedure.
    The preprocessing is the same as in TopiCLEAR.
    Afterward, KMeans and LDA are executed one after the other until the labels do not change anymore.
    KMeans always takes place in the rotated subspace.

    The rotation_ attribute contains the final LDA rotation.
    The prediction_rotation_ attribute contains the rotation used to fit the final KMeans model.
    These can differ when max_iter is reached.

    Parameters
    ----------
    n_clusters : int
        the number of clusters
    n_dims : int
        The number of features in the resulting subspace. If None this will be equal to n_clusters - 1 (default: None)
    max_iter : int
        the maximum number of iterations (default: 10)
    n_init : int
        number of times the procedure is executed using different seeds. The final result will be the one with lowest costs (default: 1)
    kmeans_repetitions : int
        Number of repetitions when executing KMeans (default: 10)
    dim_preprocess : int
        PCA dimensionality used in the preprocessing stage before normalization (default: 64).
    random_state_pca : int | None
        Random seed passed to the preprocessing PCA (default: 50).
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)
    input_type : str
        The type of input data provided. Can be either "embedding" for pre-computed embeddings or "text" for raw text input. (default: "text")
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
        The final LDA rotation
    prediction_rotation_ : np.ndarray
        The rotation used to fit the final KMeans model and used by predict
    cluster_centers_ : np.ndarray
        The cluster centers of the final KMeans model in the prediction subspace
    error_ : float
        The inertia of the final KMeans model
    kmeans_ : KMeans
        The final fitted KMeans model
    n_features_in_ : int
        The number of features used for fitting

    References
    ----------
    Ding, Chris, and Tao Li. "Adaptive dimension reduction using discriminant analysis and k-means clustering."
    Proceedings of the 24th international conference on Machine learning. 2007.
    """

    def __init__(self, n_clusters: int, n_dims: int = None, max_iter: int = 10, n_init: int = 1,
                 kmeans_repetitions: int = 10, random_state: np.random.RandomState | int = None,
                 dim_preprocess: int = 64, random_state_pca: int | None = 50,
                 input_type: str = "text", embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embedding_model: Any | None = None, embedding_batch_size: int = 64,
                 embedding_device: str | None = None, embedding_show_progress_bar: bool = False):
        super().__init__(
            n_clusters=n_clusters,
            n_dims=n_dims,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
            dim_preprocess=dim_preprocess,
            random_state_pca=random_state_pca,
            input_type=input_type,
            embedding_model_name=embedding_model_name,
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            embedding_device=embedding_device,
            embedding_show_progress_bar=embedding_show_progress_bar
        )
        self.kmeans_repetitions = kmeans_repetitions

    def _fit_on_array(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "TopiCLEARKMeans":
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels are contained in the labels_ attribute.
        """
        self._fit_preprocessor(X)
        normalized_vectors = self._preprocess(X)

        all_random_states = self.random_state.choice(
            10000,
            self.n_init,
            replace=False
        )

        best_costs = np.inf

        for i in range(self.n_init):
            local_random_state = check_random_state(
                all_random_states[i]
            )

            (
                labels,
                rotation,
                cluster_centers,
                error,
                kmeans,
                prediction_rotation
            ) = _topiclear_kmeans(
                normalized_vectors,
                self.n_clusters,
                self.n_dims,
                self.max_iter,
                self.kmeans_repetitions,
                local_random_state
            )

            if error < best_costs:
                best_costs = error

                self.labels_ = labels
                self.rotation_ = rotation
                self.prediction_rotation_ = prediction_rotation
                self.cluster_centers_ = cluster_centers
                self.error_ = error
                self.kmeans_ = kmeans

        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X: Sequence[str] | np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new input data using the final fitted KMeans model.
        """
        X_arr = self._ensure_array(X)

        check_is_fitted(
            self,
            [
                "prediction_rotation_",
                "kmeans_",
                "preprocess_pca_"
            ]
        )

        normalized_vectors = self._preprocess(X_arr)

        X_prediction_subspace = np.matmul(
            normalized_vectors,
            self.prediction_rotation_
        )

        return self.kmeans_.predict(
            X_prediction_subspace
        )

    def predict_proba(self, X: Sequence[str] | np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "TopiCLEARKMeans does not provide predict_proba because KMeans does not produce probabilistic assignments."
        )

    def score(self, X: Sequence[str] | np.ndarray) -> float:
        raise NotImplementedError(
            "TopiCLEARKMeans does not provide the GMM likelihood score implemented by TopiCLEAR."
        )
