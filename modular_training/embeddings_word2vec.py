"""
Word2Vec embeddings for multilabel text classification.
Trains Word2Vec on corpus and generates document-level embeddings by averaging word vectors.
"""

import json
import os
import pickle
from typing import List, Tuple

import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from .data_utils import preprocess_for_tfidf


class Word2VecVectorizer:
    """
    Word2Vec-based text vectorizer for multilabel classification.
    
    Features:
    - Trains Word2Vec on corpus (skip-gram or CBOW)
    - Generates document embeddings by averaging word vectors
    - Handles out-of-vocabulary words gracefully
    - Serializable for model persistence
    """
    
    def __init__(
        self,
        vector_size: int = 300,
        window: int = 5,
        min_count: int = 2,
        sg: int = 1,  # 1=skip-gram, 0=CBOW
        workers: int = 4,
        seed: int = 42,
    ):
        """
        Initialize Word2Vec vectorizer.
        
        Args:
            vector_size: Dimension of word vectors
            window: Context window size
            min_count: Minimum word frequency
            sg: Skip-gram (1) or CBOW (0)
            workers: Number of worker threads
            seed: Random seed for reproducibility
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.workers = workers
        self.seed = seed
        self.model = None
        self.oov_vector = None  # Out-of-vocabulary handling
        
    def fit(self, texts: List[str]) -> "Word2VecVectorizer":
        """
        Train Word2Vec model on corpus.
        
        Args:
            texts: List of text documents
            
        Returns:
            self for method chaining
        """
        # Preprocess texts: tokenize and clean
        sentences = [
            simple_preprocess(preprocess_for_tfidf(text))
            for text in texts
        ]
        
        # Train Word2Vec
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            workers=self.workers,
            seed=self.seed,
            epochs=5,
        )
        
        # OOV vector: random vector for unknown words
        np.random.seed(self.seed)
        self.oov_vector = np.random.randn(self.vector_size).astype(np.float32)
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Convert texts to document embeddings.
        
        Document embedding = mean of all word vectors in the document.
        
        Args:
            texts: List of text documents
            
        Returns:
            Document embeddings (n_samples, vector_size)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        embeddings = []
        
        for text in texts:
            # Preprocess and tokenize
            tokens = simple_preprocess(preprocess_for_tfidf(text))
            
            if not tokens:
                # Empty document: use zero vector
                embeddings.append(np.zeros(self.vector_size, dtype=np.float32))
            else:
                # Get vectors for all tokens
                vectors = []
                for token in tokens:
                    if token in self.model.wv:
                        vectors.append(self.model.wv[token])
                    else:
                        vectors.append(self.oov_vector)
                
                # Average vectors to get document embedding
                doc_embedding = np.mean(vectors, axis=0).astype(np.float32)
                embeddings.append(doc_embedding)
        
        return np.array(embeddings, dtype=np.float32)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit model and transform texts in one step.
        
        Args:
            texts: List of text documents
            
        Returns:
            Document embeddings (n_samples, vector_size)
        """
        return self.fit(texts).transform(texts)
    
    def save(self, path: str) -> None:
        """
        Save vectorizer and Word2Vec model.
        
        Args:
            path: Directory to save files
        """
        os.makedirs(path, exist_ok=True)
        
        # Save Word2Vec model
        self.model.save(os.path.join(path, "word2vec_model.bin"))
        
        # Save vectorizer config
        config = {
            "vector_size": self.vector_size,
            "window": self.window,
            "min_count": self.min_count,
            "sg": self.sg,
            "seed": self.seed,
        }
        with open(os.path.join(path, "vectorizer_config.json"), "w") as f:
            json.dump(config, f)
        
        # Save OOV vector
        np.save(os.path.join(path, "oov_vector.npy"), self.oov_vector)
    
    @staticmethod
    def load(path: str) -> "Word2VecVectorizer":
        """
        Load vectorizer and Word2Vec model.
        
        Args:
            path: Directory containing saved files
            
        Returns:
            Word2VecVectorizer instance
        """
        # Load config
        with open(os.path.join(path, "vectorizer_config.json"), "r") as f:
            config = json.load(f)
        
        # Create vectorizer
        vectorizer = Word2VecVectorizer(**config)
        
        # Load Word2Vec model
        vectorizer.model = Word2Vec.load(os.path.join(path, "word2vec_model.bin"))
        
        # Load OOV vector
        vectorizer.oov_vector = np.load(os.path.join(path, "oov_vector.npy"))
        
        return vectorizer


def create_word2vec_embeddings(
    texts: List[str],
    vector_size: int = 300,
    window: int = 5,
    min_count: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, Word2VecVectorizer]:
    """
    Quick utility to create Word2Vec embeddings from texts.
    
    Args:
        texts: List of text documents
        vector_size: Embedding dimension
        window: Context window size
        min_count: Minimum word frequency
        seed: Random seed
        
    Returns:
        (embeddings, vectorizer): Document embeddings and fitted vectorizer
    """
    vectorizer = Word2VecVectorizer(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        seed=seed,
    )
    embeddings = vectorizer.fit_transform(texts)
    return embeddings, vectorizer
