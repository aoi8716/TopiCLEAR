import numpy as np
from clustpy.partition import LDAKmeans
from sklearn.decomposition import PCA

from topiclear import TopiCLEARKMeans


def test_kmeans_defaults_match_paper_protocol():
    model = TopiCLEARKMeans(
        n_clusters=4,
        input_type="embedding",
        random_state=0,
    )

    assert model.n_dims == 3
    assert model.max_iter == 10
    assert model.n_init == 1
    assert model.kmeans_repetitions == 10
    assert model.dim_preprocess == 64
    assert model.random_state_pca == 50


def test_kmeans_matches_direct_ldakmeans_protocol():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 80)

    pca = PCA(
        n_components=64,
        random_state=50,
    )
    reduced_vectors = pca.fit_transform(X)

    norm_coef = np.sqrt(
        np.sum(
            reduced_vectors * reduced_vectors,
            axis=1,
        ).reshape(-1, 1)
    )
    normalized_vectors = reduced_vectors / norm_coef

    reference = LDAKmeans(
        n_clusters=4,
        n_dims=3,
        max_iter=10,
        n_init=1,
        kmeans_repetitions=10,
        random_state=7,
    )
    reference.fit(normalized_vectors)

    model = TopiCLEARKMeans(
        n_clusters=4,
        n_dims=3,
        max_iter=10,
        n_init=1,
        kmeans_repetitions=10,
        dim_preprocess=64,
        random_state_pca=50,
        input_type="embedding",
        random_state=7,
    )
    model.fit(X)

    assert np.array_equal(
        reference.labels_,
        model.labels_,
    )


def test_kmeans_matches_direct_ldakmeans_with_multiple_initializations():
    rng = np.random.RandomState(1)
    X = rng.randn(150, 80)

    pca = PCA(
        n_components=64,
        random_state=50,
    )
    reduced_vectors = pca.fit_transform(X)

    norm_coef = np.sqrt(
        np.sum(
            reduced_vectors * reduced_vectors,
            axis=1,
        ).reshape(-1, 1)
    )
    normalized_vectors = reduced_vectors / norm_coef

    reference = LDAKmeans(
        n_clusters=4,
        n_dims=3,
        max_iter=3,
        n_init=2,
        kmeans_repetitions=10,
        random_state=7,
    )
    reference.fit(normalized_vectors)

    model = TopiCLEARKMeans(
        n_clusters=4,
        n_dims=3,
        max_iter=3,
        n_init=2,
        kmeans_repetitions=10,
        dim_preprocess=64,
        random_state_pca=50,
        input_type="embedding",
        random_state=7,
    )
    model.fit(X)

    assert np.array_equal(
        reference.labels_,
        model.labels_,
    )
