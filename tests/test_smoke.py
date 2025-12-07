def test_imports():
    from topiclear import TopiCLEAR
    model = TopiCLEAR(n_clusters=3, input_type="embedding")
    assert model.n_clusters == 3
