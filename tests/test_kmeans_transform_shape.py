import numpy as np
from topiclear import TopiCLEARKMeans


def test_kmeans_transform_output_shape_matches_n_dims():
    rng = np.random.RandomState(0)
    X = rng.randn(80, 32)

    model = TopiCLEARKMeans(
        n_clusters=4,
        n_dims=3,
        input_type="embedding",
        random_state=0,
    )

    model.fit(X)
    X_transformed = model.transform(X)

    assert X_transformed.shape == (80, 3)
