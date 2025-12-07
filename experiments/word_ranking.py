import collections
from collections.abc import Sequence
from typing import Hashable, Dict, List, Tuple, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def make_wordranking_tokens(tokens_list: Sequence[Sequence[str]], predicted_labels: Sequence[int], topic_num: int, topn: int = 20,
                            ) -> Dict[int, List[tuple[str, int]]]:
    """
    Create a word ranking for each topic based on token frequencies.
    
    Parameters
    ----------
    tokens_list : Sequence[Sequence[str]]
        A sequence of tokenized documents.
    predicted_labels : Sequence[int]
        A sequence of predicted topic labels for each document.
    topic_num : int
        The number of topics.
    topn : int, optional
        The number of top words to return for each topic, by default 20.
        
    Returns
    -------
    Dict[int, List[tuple[str, int]]]
        A dictionary mapping each topic to a list of tuples containing the top words and their frequencies.
    """
    
    word_ranking: Dict[int, List[tuple[str, int]]] = {}
    for topic in range(topic_num):
        topic_tokens = [
            tokens_list[i]
            for i in range(len(predicted_labels))
            if predicted_labels[i] == topic
        ]
        words = [w for doc in topic_tokens for w in doc]
        word_counts = collections.Counter(words)
        word_ranking[topic] = word_counts.most_common(topn)
    return word_ranking


def _encode_labels_preserve_order(
    labels: Sequence[Hashable],
) -> Tuple[np.ndarray, List[Hashable]]:
    """
    Encode labels as integers while preserving their first occurrence order.
    """
    idx: Dict[Hashable, int] = {}
    enc: List[int] = []
    order: List[Hashable] = []

    for y in labels:
        if y not in idx:
            idx[y] = len(order)
            order.append(y)
        enc.append(idx[y])

    return np.array(enc, dtype=int), order

def label_specific_topwords_delta_tfidf(tokens_list: Sequence[Sequence[str]], labels: Sequence[Hashable], topn: int = 20, min_df: int = 5, vectorizer: TfidfVectorizer | None = None,
                                        ) -> Dict[Hashable, List[str]]:
    """
    Identify top words specific to each label using delta TF-IDF.
    
    Parameters
    ----------
    tokens_list : Sequence[Sequence[str]]
        A sequence of tokenized documents.
    labels : Sequence[Hashable]
        A sequence of labels corresponding to each document.
    topn : int, optional
        The number of top words to return for each label, by default 20.
    min_df : int, optional
        The minimum document frequency for the TF-IDF vectorizer, by default 5.
    vectorizer : TfidfVectorizer | None, optional
        An optional pre-initialized TF-IDF vectorizer. If None, a new one will be created and fitted, by default None.
        
    Returns
    -------
    Dict[Hashable, List[str]]
        A dictionary mapping each label to a list of its top words.
    """
    if len(tokens_list) == 0:
        raise ValueError("tokens_list is empty.")

    if len(tokens_list) != len(labels):
        raise ValueError("tokens_list and labels must have the same length.")

    enc_labels, label_order = _encode_labels_preserve_order(labels)

    docs = [" ".join(tokens) for tokens in tokens_list]

    if vectorizer is None:
        vectorizer = TfidfVectorizer(min_df=min_df)
        X = vectorizer.fit_transform(docs)
    else:
        X = vectorizer.transform(docs)

    feature_names = np.array(vectorizer.get_feature_names_out())
    L = len(label_order)
    label2words: Dict[Hashable, List[str]] = {}

    for i, lab in enumerate(label_order):
        mask_lab = enc_labels == i
        mask_other = ~mask_lab

        if not np.any(mask_lab):
            label2words[lab] = []
            continue

        X_lab = X[mask_lab]
        X_other = X[mask_other] if np.any(mask_other) else None

        mean_lab = np.asarray(X_lab.mean(axis=0)).ravel()
        if X_other is None:
            mean_other = np.zeros_like(mean_lab)
        else:
            mean_other = np.asarray(X_other.mean(axis=0)).ravel()

        score = mean_lab - mean_other  # delta TF-IDF

        top_idx = np.argsort(score)[::-1][:topn]
        label2words[lab] = feature_names[top_idx].tolist()

    return label2words