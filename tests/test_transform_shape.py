import numpy as np
from topiclear import TopiCLEAR

def test_transform_output_shape_matches_n_dims():
    rng = np.random.RandomState(0)
    X = rng.randn(40, 32) # 40 docs, 32-dim embeddings
    
    model = TopiCLEAR(
        n_clusters=4,
        n_dims=3, 
        input_type="embedding",
        random_state=0,
    )

    model.fit(X)
    X_transformed = model.transform(X)

    # Check shape
    assert X_transformed.shape == (40, 3)