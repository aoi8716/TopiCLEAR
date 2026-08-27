import numpy as np

from topiclear import TopiCLEARKMeans


def test_kmeans_predict_matches_training_labels_when_max_iter_is_reached():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 80)

    model = TopiCLEARKMeans(
        n_clusters=4,
        n_dims=3,
        max_iter=1,
        input_type="embedding",
        random_state=7,
    )

    model.fit(X)
    predicted_labels = model.predict(X)

    assert np.array_equal(
        model.labels_,
        predicted_labels
    )

    assert model.prediction_rotation_.shape == (
        64,
        3,
    )
