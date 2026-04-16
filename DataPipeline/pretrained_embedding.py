# coding=utf-8
"""
Pretrained Embedding Module
This module provides classes for loading and initializing pretrained word embeddings.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from .vocabulary import Vocabulary, SpecialTokens


class EmbeddingLoadResult:
    """
    Container for pretrained embedding loading results.
    
    Attributes:
        embedding_matrix: Tensor of shape (vocab_size, embedding_dim)
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of the embeddings
        iv_count: Number of in-vocabulary words found in pretrained embeddings
        oov_count: Number of out-of-vocabulary words
        oov_ratio: Ratio of OOV words
    """
    
    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        vocab_size: int,
        embedding_dim: int,
        iv_count: int,
        oov_count: int
    ):
        self.embedding_matrix = embedding_matrix
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.iv_count = iv_count
        self.oov_count = oov_count
        self.oov_ratio = oov_count / vocab_size if vocab_size > 0 else 0.0
    
    def __repr__(self) -> str:
        return (f"EmbeddingLoadResult(vocab_size={self.vocab_size}, "
                f"embedding_dim={self.embedding_dim}, "
                f"iv_count={self.iv_count}, "
                f"oov_count={self.oov_count}, "
                f"oov_ratio={self.oov_ratio:.4f})")


class OOVStrategy(ABC):
    """
    Abstract base class for out-of-vocabulary word initialization strategies.
    """
    
    @abstractmethod
    def initialize(
        self,
        embedding_matrix: np.ndarray,
        oov_indices: List[int],
        vocab: Vocabulary,
        iv_embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Initialize embeddings for OOV words.
        
        Args:
            embedding_matrix: Current embedding matrix
            oov_indices: List of indices for OOV words
            vocab: Vocabulary instance
            iv_embeddings: Embeddings of in-vocabulary words (for avg strategy)
            
        Returns:
            Updated embedding matrix
        """
        pass


class ZeroOOVStrategy(OOVStrategy):
    """
    OOV strategy that initializes OOV words to zero vectors.
    """
    
    def initialize(
        self,
        embedding_matrix: np.ndarray,
        oov_indices: List[int],
        vocab: Vocabulary,
        iv_embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Initialize OOV words to zero vectors."""
        # Already zeros, just return
        return embedding_matrix


class UniformOOVStrategy(OOVStrategy):
    """
    OOV strategy that initializes OOV words with uniform distribution.
    """
    
    def __init__(self, low: float = -0.25, high: float = 0.25, seed: int = 233):
        """
        Initialize the uniform OOV strategy.
        
        Args:
            low: Lower bound of uniform distribution
            high: Upper bound of uniform distribution
            seed: Random seed
        """
        self.low = low
        self.high = high
        self.seed = seed
        self._rng = np.random.RandomState(seed)
    
    def initialize(
        self,
        embedding_matrix: np.ndarray,
        oov_indices: List[int],
        vocab: Vocabulary,
        iv_embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Initialize OOV words with uniform distribution."""
        embedding_dim = embedding_matrix.shape[1]
        
        # Generate uniform vectors for all OOV words (excluding pad)
        uniform_vector = self._rng.uniform(
            low=self.low, 
            high=self.high, 
            size=embedding_dim
        ).round(6)
        
        for idx in oov_indices:
            if idx != vocab.pad_idx:
                embedding_matrix[idx] = uniform_vector
        
        return embedding_matrix


class AverageOOVStrategy(OOVStrategy):
    """
    OOV strategy that initializes OOV words with the average of in-vocabulary embeddings.
    """
    
    def initialize(
        self,
        embedding_matrix: np.ndarray,
        oov_indices: List[int],
        vocab: Vocabulary,
        iv_embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Initialize OOV words with average of in-vocabulary embeddings."""
        if iv_embeddings is None or len(iv_embeddings) == 0:
            # No IV embeddings, use zeros
            return embedding_matrix
        
        # Calculate average of in-vocabulary embeddings
        avg_embedding = np.mean(iv_embeddings, axis=0)
        
        # Apply to OOV words (excluding pad)
        for idx in oov_indices:
            if idx != vocab.pad_idx:
                embedding_matrix[idx] = avg_embedding
        
        return embedding_matrix


class XavierOOVStrategy(OOVStrategy):
    """
    OOV strategy that initializes OOV words with Xavier uniform initialization.
    """
    
    def __init__(self, seed: int = 233):
        """
        Initialize the Xavier OOV strategy.
        
        Args:
            seed: Random seed
        """
        self.seed = seed
    
    def initialize(
        self,
        embedding_matrix: np.ndarray,
        oov_indices: List[int],
        vocab: Vocabulary,
        iv_embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Initialize OOV words with Xavier uniform initialization."""
        vocab_size, embedding_dim = embedding_matrix.shape
        
        # Create a temporary embedding layer for Xavier initialization
        embed_layer = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(embed_layer.weight.data)
        
        # Get the initialized weights
        xavier_weights = embed_layer.weight.data.numpy()
        
        # Apply to OOV words (excluding pad)
        for idx in oov_indices:
            if idx != vocab.pad_idx:
                embedding_matrix[idx] = xavier_weights[idx]
        
        return embedding_matrix


class PretrainedEmbeddingLoader:
    """
    Loader for pretrained word embeddings.
    
    Handles:
    - Reading pretrained embedding files
    - Matching embeddings to vocabulary
    - Initializing OOV words with various strategies
    """
    
    def __init__(
        self,
        file_path: str,
        oov_strategy: OOVStrategy,
        verbose: bool = True
    ):
        """
        Initialize the pretrained embedding loader.
        
        Args:
            file_path: Path to the pretrained embedding file
            oov_strategy: Strategy for initializing OOV words
            verbose: Whether to print progress information
        """
        self.file_path = file_path
        self.oov_strategy = oov_strategy
        self.verbose = verbose
        
        # Cache for pretrained embeddings
        self._pretrained_cache: Optional[Dict[str, np.ndarray]] = None
        self._embedding_dim: Optional[int] = None
    
    def _load_pretrained_file(self) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Load pretrained embeddings from file.
        
        Returns:
            A tuple of (word_to_embedding_dict, embedding_dim)
        """
        if self._pretrained_cache is not None:
            return self._pretrained_cache, self._embedding_dim
        
        if self.verbose:
            print(f"Loading pretrained embeddings from: {self.file_path}")
        
        word_to_embedding: Dict[str, np.ndarray] = {}
        embedding_dim = -1
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(' ')
                
                # Check for header line (word2vec format: first line is vocab_size, dim)
                if len(parts) == 2:
                    try:
                        vocab_size = int(parts[0])
                        embedding_dim = int(parts[1])
                        continue
                    except ValueError:
                        pass
                
                if len(parts) <= 1:
                    continue
                
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                
                if embedding_dim == -1:
                    embedding_dim = len(vector)
                
                word_to_embedding[word] = vector
        
        self._pretrained_cache = word_to_embedding
        self._embedding_dim = embedding_dim
        
        if self.verbose:
            print(f"Loaded {len(word_to_embedding)} pretrained embeddings with dimension {embedding_dim}")
        
        return word_to_embedding, embedding_dim
    
    def load_for_vocab(self, vocab: Vocabulary) -> EmbeddingLoadResult:
        """
        Load pretrained embeddings for a given vocabulary.
        
        Args:
            vocab: Vocabulary instance
            
        Returns:
            An EmbeddingLoadResult instance
        """
        # Load pretrained embeddings
        pretrained_embeddings, embedding_dim = self._load_pretrained_file()
        
        vocab_size = len(vocab)
        
        # Initialize embedding matrix with zeros
        embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
        
        # Track in-vocabulary and out-of-vocabulary words
        iv_count = 0
        oov_indices: List[int] = []
        iv_embeddings: List[np.ndarray] = []
        
        # Match vocabulary words with pretrained embeddings
        for word, idx in vocab.stoi.items():
            if word in pretrained_embeddings:
                embedding_matrix[idx] = pretrained_embeddings[word]
                iv_count += 1
                iv_embeddings.append(pretrained_embeddings[word])
            else:
                oov_indices.append(idx)
        
        # Initialize OOV words
        iv_embeddings_np = np.array(iv_embeddings) if iv_embeddings else None
        embedding_matrix = self.oov_strategy.initialize(
            embedding_matrix,
            oov_indices,
            vocab,
            iv_embeddings_np
        )
        
        oov_count = len(oov_indices)
        
        if self.verbose:
            print(f"Vocabulary size: {vocab_size}")
            print(f"In-vocabulary words found: {iv_count}")
            print(f"Out-of-vocabulary words: {oov_count}")
            print(f"OOV ratio: {oov_count / vocab_size:.4f}")
        
        # Convert to PyTorch tensor
        embedding_tensor = torch.from_numpy(embedding_matrix).float()
        
        return EmbeddingLoadResult(
            embedding_matrix=embedding_tensor,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            iv_count=iv_count,
            oov_count=oov_count
        )


class EmbeddingInitializer:
    """
    Factory class for creating embedding loaders with different strategies.
    """
    
    STRATEGY_ZERO = 'zero'
    STRATEGY_UNIFORM = 'uniform'
    STRATEGY_AVERAGE = 'average'
    STRATEGY_XAVIER = 'xavier'
    
    @classmethod
    def create_loader(
        cls,
        file_path: str,
        strategy: str = 'average',
        verbose: bool = True,
        **kwargs
    ) -> PretrainedEmbeddingLoader:
        """
        Create a pretrained embedding loader with the specified strategy.
        
        Args:
            file_path: Path to the pretrained embedding file
            strategy: OOV initialization strategy ('zero', 'uniform', 'average', 'xavier')
            verbose: Whether to print progress information
            **kwargs: Additional arguments for the strategy
            
        Returns:
            A PretrainedEmbeddingLoader instance
        """
        if strategy == cls.STRATEGY_ZERO:
            oov_strategy = ZeroOOVStrategy()
        elif strategy == cls.STRATEGY_UNIFORM:
            low = kwargs.get('low', -0.25)
            high = kwargs.get('high', 0.25)
            seed = kwargs.get('seed', 233)
            oov_strategy = UniformOOVStrategy(low=low, high=high, seed=seed)
        elif strategy == cls.STRATEGY_AVERAGE:
            oov_strategy = AverageOOVStrategy()
        elif strategy == cls.STRATEGY_XAVIER:
            seed = kwargs.get('seed', 233)
            oov_strategy = XavierOOVStrategy(seed=seed)
        else:
            raise ValueError(f"Unknown OOV strategy: {strategy}")
        
        return PretrainedEmbeddingLoader(
            file_path=file_path,
            oov_strategy=oov_strategy,
            verbose=verbose
        )
