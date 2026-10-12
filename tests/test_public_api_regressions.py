import numpy as np

from experiments.methods.topiclear_runner import run_topiclear
from topiclear import TopiCLEAR, TopiCLEARKMeans


def make_embeddings(n_samples=180, n_features=80):
    return np.random.RandomState(12).normal(size=(n_samples, n_features))


def test_runner_default_uses_adr_dimensions():
    X = make_embeddings(n_samples=120, n_features=80)
    model, labels = run_topiclear(
        X, n_topics=3, input_type="embedding",
        max_iter=2, gmm_repetitions=1, random_state=7,
    )
    assert model.n_dims == 2
    assert model.rotation_.shape == (64, 2)
    assert model.n_iter_ >= 1
    assert labels.shape == (120,)


def test_integer_seed_repeats_fitting_in_the_same_gmm_estimator():
    X = make_embeddings()
    model = TopiCLEAR(
        n_clusters=3, input_type="embedding", random_state=42,
        max_iter=2, n_init=1, gmm_repetitions=1,
    )
    assert model.random_state == 42
    model.fit(X)
    labels = model.labels_.copy()
    rotation = model.rotation_.copy()
    error = model.error_
    model.fit(X)
    np.testing.assert_array_equal(model.labels_, labels)
    np.testing.assert_allclose(model.rotation_, rotation, rtol=0, atol=0)
    assert model.error_ == error


def test_inner_gmm_pca_does_not_depend_on_global_numpy_seed():
    # n_samples > 500 and n_dims < 80% of min(shape) can trigger randomized PCA.
    X = make_embeddings(n_samples=530, n_features=80)
    original_global_state = np.random.get_state()
    try:
        np.random.seed(1)
        first = TopiCLEAR(
            n_clusters=3, input_type="embedding", random_state=42,
            max_iter=1, n_init=1, gmm_repetitions=1,
        ).fit(X)
        np.random.seed(999)
        second = TopiCLEAR(
            n_clusters=3, input_type="embedding", random_state=42,
            max_iter=1, n_init=1, gmm_repetitions=1,
        ).fit(X)
    finally:
        np.random.set_state(original_global_state)
    np.testing.assert_array_equal(first.labels_, second.labels_)
    np.testing.assert_allclose(first.rotation_, second.rotation_, rtol=0, atol=0)


def test_score_is_gmm_mean_log_likelihood_not_loss():
    X = make_embeddings()
    model = TopiCLEAR(
        n_clusters=3, input_type="embedding", random_state=42,
        max_iter=2, gmm_repetitions=1,
    ).fit(X)
    expected = model.gmm_.score(model.transform(X))
    assert np.isclose(model.score(X), expected)
    assert np.isclose(model.score(X), -model.error_)


def test_integer_seed_repeats_fitting_in_the_same_kmeans_estimator():
    # The small dataset takes sklearn's non-randomized PCA path. The paper's
    # unseeded ClustPy inner PCA itself is intentionally not modified here.
    X = make_embeddings(n_samples=160)
    model = TopiCLEARKMeans(
        n_clusters=3, input_type="embedding", random_state=42,
        max_iter=2, n_init=1, kmeans_repetitions=2,
    )
    assert model.random_state == 42
    model.fit(X)
    labels = model.labels_.copy()
    model.fit(X)
    np.testing.assert_array_equal(model.labels_, labels)
