import numpy as np

from topiclear import TopiCLEARKMeans


def test_kmeans_random_state_is_reproducible():
    rng = np.random.RandomState(0)
    X = rng.randn(120, 80)

    model1 = TopiCLEARKMeans(
        n_clusters=4,
        input_type="embedding",
        random_state=7,
    )

    model2 = TopiCLEARKMeans(
        n_clusters=4,
        input_type="embedding",
        random_state=7,
    )

    model1.fit(X)
    model2.fit(X)

    assert np.array_equal(
        model1.labels_,
        model2.labels_,
    )
