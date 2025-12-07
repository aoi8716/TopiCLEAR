import numpy as np
from topiclear import TopiCLEAR

class DummyModel:
    def __init__(self, dim: int = 8):
        self.dim = dim

    def encode(self, texts, batch_size=64, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=False):
        rng = np.random.RandomState(0)
        return rng.randn(len(texts), self.dim)

def test_text_input_small_sample():
    texts = ["hello", "world", "test", "text", "input"]
    model = TopiCLEAR(
        n_clusters=2,
        input_type="text",
        embedding_model=DummyModel(dim=16),
        random_state=0
    )
    labels = model.fit_predict(texts)
    assert labels.shape == (len(texts),)