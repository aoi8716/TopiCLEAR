import pytest
import numpy as np
from topiclear import TopiCLEAR

def test_error_when_numpy_with_text_input_type():
    X = np.random.randn(10, 8)
    model = TopiCLEAR(n_clusters=3, input_type="text")
    with pytest.raises(ValueError):
        model.fit(X)