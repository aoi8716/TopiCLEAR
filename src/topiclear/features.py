from typing import Sequence, Optional, Any
import numpy as np

def build_sentence_embeddings(
    texts: Sequence[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    model: Optional[Any] = None,
    batch_size: int = 64,
    show_progress_bar: bool = False,
    device: Optional[str] = None
) -> np.ndarray:
    
    """
    Generate sentence embeddings for a list of texts using a pre-trained SentenceTransformer model.

    Parameters
    ----------
    texts : Sequence[str]
        A sequence of text strings to be embedded.
    model_name : str, optional
        The name of the pre-trained SentenceTransformer model to use, by default "sentence-transformers/all-MiniLM-L6-v2".
    model : Optional[Any], optional
        An optional pre-initialized SentenceTransformer model. If provided, this model will be used instead of loading a new one, by default None.
    batch_size : int, optional
        The batch size for processing texts, by default 64.
    show_progress_bar : bool, optional
        Whether to display a progress bar during embedding, by default False.
    device : Optional[str], optional
        The device to run the model on (e.g., "cpu" or "cuda"), by default None.
        
    Returns
    -------
    np.ndarray
        An array of shape (n_texts, embedding_dimension) containing the sentence embeddings.
    """
    
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("Please install the 'sentence-transformers' package to use this function.") from e
        model = SentenceTransformer(model_name, device=device)

    vectors = model.encode(list(texts), batch_size=batch_size, show_progress_bar=show_progress_bar, convert_to_numpy=True, normalize_embeddings=False)
    return vectors