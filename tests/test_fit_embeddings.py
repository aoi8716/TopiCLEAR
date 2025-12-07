import numpy as np
from topiclear import TopiCLEAR

def test_fit_predict_on_embeddings_returns_labels():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 16)  # 50 docs, 16-dim embeddings

    model = TopiCLEAR(
        n_clusters=3,
        input_type="embedding",
        random_state=0,
    )

    labels = model.fit_predict(X)

    # Check shape
    assert labels.shape == (50,)

    # Check label range
    assert labels.min() >= 0
    assert labels.max() < 3

    # Check number of unique clusters
    unique = np.unique(labels)
    assert 1 <= len(unique) <= 3