from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional

def load_texts_from_csv(path: str | Path, text_column: str = "text") -> list[str]:
    df = pd.read_csv(path)
    return df[text_column].astype(str).tolist()

def load_texts_and_labels_from_csv(path: str | Path, text_column: str = "text", label_column: str = "label"
                                ) -> Tuple[list[str], Optional[np.ndarray]]:
    df = pd.read_csv(path)
    texts = df[text_column].astype(str).tolist()
    if label_column in df.columns:
        labels = df[label_column].tolist()
        return texts, np.array(labels)
    else:
        return texts, None