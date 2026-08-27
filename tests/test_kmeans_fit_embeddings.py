import numpy as np

from topiclear import TopiCLEARKMeans


def test_kmeans_fit_predict_on_embeddings_returns_labels():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 16)

    model = TopiCLEARKMeans(
        n_clusters=3,
        input_type="embedding",
        random_state=0,
    )

    labels = model.fit_predict(X)

    assert labels.shape == (50,)
    assert np.array_equal(labels, model.labels_)
