from dataclasses import dataclass
from collections import Counter
from typing import Sequence, Iterable, Optional

import re

from nltk import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from stop_words import get_stop_words


@dataclass
class BoWPreprocessConfig:
    min_token_len: int = 2
    min_global_freq: int = 5
    remove_non_alnum: bool = True
    
    # extra stopwords
    extra_stopwords: Optional[Sequence[str]] = None
    extra_stop_chars: Optional[Sequence[str]] = None
    
class BoWPreprocessor:
    def __init__(self, config: Optional[BoWPreprocessConfig] = None) -> None:
        if config is None:
            config = BoWPreprocessConfig()
        self.config = config
        self._stop_words = self._build_stopwords()
        
    def transform(self, texts: Sequence[str]) -> list[list[str]]:
        tokenized_nonstop = [self._tokenize(text) for text in texts]
        freq = Counter(token for doc in tokenized_nonstop for token in doc)
        
        rare_words = {word for word, count in freq.items() if count <= self.config.min_global_freq}
        
        irregular_words: set[str] = set()
        if self.config.remove_non_alnum:
            for word in freq.keys():
                if len(re.sub(r"[^a-zA-Z0-9]", "", word)) < 1:
                    irregular_words.add(word)
        
        exclude = rare_words | irregular_words
        
        processed = [
            [token for token in doc if token not in exclude]
            for doc in tokenized_nonstop
        ]
        return processed
    
    def _build_stopwords(self) -> set[str]:
        cfg = self.config
        
        stop_words = set(nltk_stopwords.words("english")) | set(get_stop_words("english"))
        
        default_extra = ["would", "could", "should"]
        default_chars = [
            "==", "--", "'s", "''", "n't", "``", "..", "...", "....",
            "'m", "'ve", "'re", "'d", "'ll", "", "-+", "+-", "_/", "||",
            "__", "/|", "//",
        ]
        
        extra = cfg.extra_stopwords if cfg.extra_stopwords is not None else default_extra
        chars = cfg.extra_stop_chars if cfg.extra_stop_chars is not None else default_chars
        
        stop_words.update(extra)
        stop_words.update(chars)   
        return stop_words
    
    def _tokenize(self, text: str) -> list[str]:
        cfg = self.config
        tokens = word_tokenize(text.lower())
        out: list[str] = []
        for token in tokens:
            token_low = token.lower()
            if token_low in self._stop_words:
                continue
            if len(token_low) < cfg.min_token_len:
                continue
            out.append(token_low)
        return out

