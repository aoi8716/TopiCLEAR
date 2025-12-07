import numpy as np
from topiclear import TopiCLEAR

def test_random_state_reproducible_on_embeddings():
    rng = np.random.RandomState(0)
    X = rng.randn(60, 16)

    model1 = TopiCLEAR(n_clusters=4, input_type="embedding", random_state=42)
    model2 = TopiCLEAR(n_clusters=4, input_type="embedding", random_state=42)

    labels1 = model1.fit_predict(X)
    labels2 = model2.fit_predict(X)

    assert np.array_equal(labels1, labels2)
