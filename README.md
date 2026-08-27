# TopiCLEAR

Adaptive clustering of sentence embeddings for topic extraction.

This repository contains the reference implementation of TopiCLEAR (Topic extraction by CLustering Embeddings with Adaptive dimensional Reduction), the method proposed in:

Fujita, A., Yamamoto, T., Nakayama, Y., & Kobayashi, R. (2027). "TopiCLEAR: Adaptive embedding clustering for interpretable topic discovery from short texts." Expert Systems with Applications, 333, 133996. https://doi.org/10.1016/j.eswa.2026.133996

TopiCLEAR clusters sentence level embeddings with an iterative procedure that alternates between:

- preprocessing with PCA
- Gaussian mixture clustering
- supervised projection based on linear discriminant analysis

The method is designed for short and informal texts such as tweets, Reddit comments, and news titles.

## Installation

Requirements

- Python 3.10 (tested with 3.10.14)
- A recent C and C++ toolchain if you want to run all baselines in the paper

Create a virtual environment and install the dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

This uses requirements.txt together with build-constraints.txt and requirements.lock to install a set of pinned versions that were used in the experiments.

If you only want to use the TopiCLEAR library and not all baselines, you can also do:

```bash
python -m pip install .
```

which relies on the dependencies declared in pyproject.toml.

## Quick start

### Clustering raw texts

```python
from topiclear import TopiCLEAR

texts = [
    "I love sushi, especially salmon nigiri.",
    "Natural language processing often depends on a good tokenizer.",
    "It is raining today so I will stay home and read.",
    "Cats enjoy sunbathing on the balcony.",
]

model = TopiCLEAR(
    n_clusters=2,
    input_type="text",  # raw texts are passed in
    random_state=0,
)

labels = model.fit_predict(texts)
print(labels)  # numpy array of shape (len(texts),)
```

By default the implementation uses the SentenceTransformer model sentence-transformers/all-MiniLM-L6-v2 to build 384 dimensional embeddings.

### Using precomputed embeddings

If you already have embeddings, set input_type to "embedding" and pass a numpy array of shape (n_samples, n_features):

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from topiclear import TopiCLEAR

texts = [...]
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = encoder.encode(texts, convert_to_numpy=True)

model = TopiCLEAR(
    n_clusters=K,
    input_type="embedding",
    random_state=0,
)

labels = model.fit_predict(embeddings)
```

You can also provide a pre initialised embedding model through the embedding_model argument when input_type is "text"; this is useful if you want to reuse the same SentenceTransformer instance across multiple runs.

### K-means variant

The K-means variant used in the paper is available as `TopiCLEARKMeans`.

```python
from topiclear import TopiCLEARKMeans

model = TopiCLEARKMeans(
    n_clusters=K,
    input_type="embedding",
    random_state=0,
)

labels = model.fit_predict(embeddings)
```

By default, this variant uses the paper settings `dim_preprocess=64`, `random_state_pca=50`, `max_iter=10`, `n_init=1`, and `kmeans_repetitions=10`.

For out-of-sample prediction, `predict` uses the rotation associated with the final fitted KMeans model. The `transform` method continues to return the final LDA subspace.

## API overview

The main entry points are `topiclear.TopiCLEAR` and `topiclear.TopiCLEARKMeans`, which follow the scikit learn estimator interface.

Constructor arguments (main ones)

- n_clusters: number of topics to extract
- n_dims: dimension of the discriminative subspace; defaults to n_clusters minus one
- max_iter: maximum number of ADR iterations
- n_init: number of random initialisations of the algorithm
- gmm_repetitions: number of initialisations inside each GaussianMixture fit for `TopiCLEAR`
- kmeans_repetitions: number of initialisations inside each KMeans fit for `TopiCLEARKMeans`
- dim_preprocess: PCA dimension used before normalisation (default 64)
- random_state_pca: random seed for the preprocessing PCA
- random_state: seed for all other random choices
- input_type: "text" or "embedding"
- embedding_model_name: SentenceTransformer model name when input_type is "text" (default "sentence-transformers/all-MiniLM-L6-v2")
- embedding_model: optional pre initialised embedding model; if given, embedding_model_name is ignored
- embedding_batch_size: batch size used when encoding texts
- embedding_device: device passed to SentenceTransformer, for example "cpu" or "cuda"
- embedding_show_progress_bar: show a progress bar during embedding of texts

Methods

- fit(X, y=None): runs TopiCLEAR on the data
- predict(X): assign clusters to new data using the fitted Gaussian mixture for `TopiCLEAR` or the final fitted KMeans model for `TopiCLEARKMeans`
- fit_predict(X, y=None): fit the model and return the cluster labels for the fitted data
- transform(X): project data into the learned low dimensional subspace

Attributes

- labels\_: cluster labels obtained on the data passed to fit or fit_predict
- rotation\_: projection matrix that maps the preprocessed embeddings into the low dimensional subspace
- cluster_centers_: cluster centers returned by the fitted clustering model
- n_features_in_: number of features seen during fitting
- error_: final objective value; average negative log likelihood for `TopiCLEAR` and KMeans inertia for `TopiCLEARKMeans`
- n_iter_: number of ADR iterations executed by `TopiCLEAR`
- prediction_rotation_: rotation used by the final KMeans model in `TopiCLEARKMeans`
- kmeans_: final fitted KMeans model in `TopiCLEARKMeans`

For more details about the algorithm, see Section 3 of the paper and Algorithm 1 in the main text.

## Repository structure

- src/topiclear
  Core implementation of the TopiCLEAR model and helper utilities
- experiments
  Utility functions for the quantitative and qualitative evaluations in the paper, including ARI and AMI computation and delta TF IDF based word ranking
- tests
  Small unit tests that check basic behaviour and text input handling
- docs
  Reserved for extended documentation

## Reproducing the paper

This repository contains helper functions that were used to generate the results in the paper, for example:

- experiments/evaluation.py for ARI and AMI
- experiments/word_ranking.py for the qualitative analysis on TweetTopic
- experiments/methods/topiclear_runner.py as a small wrapper around the TopiCLEAR class

A complete one command script to reproduce all tables and figures is not included. Instead, we recommend the following workflow:

1. Prepare each dataset as an input file containing texts and their human annotated labels.
2. Use the TopiCLEAR class or the run_topiclear function to obtain predicted labels.
3. Evaluate ARI and AMI with the functions in experiments/evaluation.py.
4. For the TweetTopic case study, compute topic specific word lists with experiments/word_ranking.py.

The exact preprocessing and hyperparameters follow the settings described in Sections 4 and 5 of the paper.

## Citation

If you use this code or the TopiCLEAR method in academic work, please cite:

Fujita, A., Yamamoto, T., Nakayama, Y., & Kobayashi, R. (2027).
TopiCLEAR: Adaptive embedding clustering for interpretable topic discovery from short texts.
_Expert Systems with Applications, 333_, 133996.
https://doi.org/10.1016/j.eswa.2026.133996

A machine readable citation is also available in the file CITATION.cff in this repository.

## License

This project is released under the MIT License. See the file LICENSE for details.
