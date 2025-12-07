from topiclear import TopiCLEAR
from typing import Sequence, Any, Literal, Optional
import numpy as np


def run_topiclear(X: Sequence[str] | np.ndarray,
                n_topics: int,
                input_type: Literal["text", "embedding"] = "text",
                random_state: int = 0,
                n_dims: int = 64,
                max_iter: int = 10,
                n_init: int = 1,
                gmm_repetitions: int = 10,
                dim_preprocess: int = 64,
                random_state_pca: int = 50,
                embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                embedding_model: Optional[Any] = None,
                embedding_batch_size: int = 64,
                embedding_device: Optional[str] = None,
                embedding_show_progress_bar: bool = False
                ) -> tuple[TopiCLEAR, np.ndarray]:
    """
    Run the TopiCLEAR model.
    
    Parameters
    ----------
    X : Sequence[str] | np.ndarray
        Input data, either as a sequence of text strings or as pre-computed embeddings.
    n_topics : int
        The number of topics to extract.
    input_type : Literal["text", "embedding"], optional
        The type of input data, by default "text".
    random_state : int, optional
        Random state for reproducibility, by default 0.
    n_dims : Optional[int], optional    
        Number of dimensions for topic embeddings, by default None.
    max_iter : int, optional
        Maximum number of iterations for the TopiCLEAR algorithm, by default 10.
    n_init : int, optional
        Number of initializations for the TopiCLEAR algorithm, by default 1.
    gmm_repetitions : int, optional
        Number of repetitions for GMM fitting, by default 10.
    dim_preprocess : int, optional
        Dimensionality for preprocessing embeddings, by default 64.
    random_state_pca : int, optional
        Random state for PCA preprocessing, by default 0.
    embedding_model_name : str, optional
        Name of the embedding model to use, by default "sentence-transformers/all-MiniLM-L6-v2".
    embedding_model : Optional[Any], optional
        Pre-initialized embedding model, by default None.
    embedding_batch_size : int, optional
        Batch size for embedding computation, by default 64.
    embedding_device : Optional[str], optional
        Device for embedding computation, by default None.  
    embedding_show_progress_bar : bool, optional
        Whether to show a progress bar during embedding computation, by default False.
        
    Returns
    -------
    TopiCLEAR
        The fitted TopiCLEAR model instance.
    np.ndarray
        The topic assignments for the input data.
    """
    
    model = TopiCLEAR(
        n_clusters=n_topics,
        n_dims=n_dims,
        max_iter=max_iter,
        n_init=n_init,
        gmm_repetitions=gmm_repetitions,
        random_state=random_state,
        dim_preprocess=dim_preprocess,
        random_state_pca=random_state_pca,
        input_type=input_type,
        embedding_model_name=embedding_model_name,
        embedding_model=embedding_model,
        embedding_batch_size=embedding_batch_size,
        embedding_device=embedding_device,
        embedding_show_progress_bar=embedding_show_progress_bar
    )
    model.fit(X)
    return model, model.labels_ 
