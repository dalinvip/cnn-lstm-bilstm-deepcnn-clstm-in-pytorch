# coding=utf-8
"""
Vocabulary Module
This module provides classes for managing vocabulary and building word indices.
"""

from collections import Counter
from typing import List, Dict, Tuple, Optional, Iterable
from abc import ABC, abstractmethod


class SpecialTokens:
    """
    Special tokens used in vocabulary.
    """
    PAD = "<pad>"
    UNK = "<unk>"
    SOS = "<sos>"
    EOS = "<eos>"
    
    @classmethod
    def all_tokens(cls) -> List[str]:
        """Return all special tokens."""
        return [cls.PAD, cls.UNK, cls.SOS, cls.EOS]


class Vocabulary:
    """
    Vocabulary class for managing word-to-index and index-to-word mappings.
    
    Attributes:
        stoi: A dictionary mapping words to indices
        itos: A list mapping indices to words
        pad_token: The padding token
        unk_token: The unknown token
        pad_idx: The index of the padding token
        unk_idx: The index of the unknown token
    """
    
    def __init__(
        self,
        stoi: Dict[str, int],
        pad_token: str = SpecialTokens.PAD,
        unk_token: str = SpecialTokens.UNK
    ):
        """
        Initialize the vocabulary.
        
        Args:
            stoi: A dictionary mapping words to indices
            pad_token: The padding token
            unk_token: The unknown token
        """
        self.stoi: Dict[str, int] = stoi
        self.itos: List[str] = [word for word, idx in sorted(stoi.items(), key=lambda x: x[1])]
        
        self.pad_token = pad_token
        self.unk_token = unk_token
        
        # Get indices for special tokens
        self.pad_idx = self.stoi.get(pad_token, 0)
        self.unk_idx = self.stoi.get(unk_token, 1)
        
        # Validate vocabulary
        self._validate()
    
    def _validate(self) -> None:
        """Validate the vocabulary structure."""
        # Check that itos and stoi are consistent
        for idx, word in enumerate(self.itos):
            if self.stoi.get(word) != idx:
                raise ValueError(f"Inconsistent vocabulary: word '{word}' at index {idx} "
                               f"has stoi value {self.stoi.get(word)}")
        
        # Check that special tokens exist
        if self.pad_token not in self.stoi:
            raise ValueError(f"Padding token '{self.pad_token}' not found in vocabulary")
        if self.unk_token not in self.stoi:
            raise ValueError(f"Unknown token '{self.unk_token}' not found in vocabulary")
    
    def __len__(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.itos)
    
    def __contains__(self, word: str) -> bool:
        """Check if a word is in the vocabulary."""
        return word in self.stoi
    
    def lookup_word(self, word: str) -> int:
        """
        Look up the index of a word.
        
        Args:
            word: The word to look up
            
        Returns:
            The index of the word, or unk_idx if not found
        """
        return self.stoi.get(word, self.unk_idx)
    
    def lookup_index(self, idx: int) -> str:
        """
        Look up the word at a given index.
        
        Args:
            idx: The index to look up
            
        Returns:
            The word at the given index
        """
        if 0 <= idx < len(self.itos):
            return self.itos[idx]
        return self.unk_token
    
    def lookup_words(self, words: List[str]) -> List[int]:
        """
        Look up the indices of multiple words.
        
        Args:
            words: A list of words to look up
            
        Returns:
            A list of indices
        """
        return [self.lookup_word(word) for word in words]
    
    def lookup_indices(self, indices: List[int]) -> List[str]:
        """
        Look up the words at multiple indices.
        
        Args:
            indices: A list of indices to look up
            
        Returns:
            A list of words
        """
        return [self.lookup_index(idx) for idx in indices]
    
    def get_itos(self) -> List[str]:
        """Return the index-to-word list."""
        return self.itos.copy()
    
    def get_stoi(self) -> Dict[str, int]:
        """Return the word-to-index dictionary."""
        return self.stoi.copy()


class VocabularyBuilder:
    """
    Builder class for creating vocabularies from text data.
    """
    
    def __init__(
        self,
        pad_token: str = SpecialTokens.PAD,
        unk_token: str = SpecialTokens.UNK,
        min_freq: int = 1,
        max_size: Optional[int] = None
    ):
        """
        Initialize the vocabulary builder.
        
        Args:
            pad_token: The padding token
            unk_token: The unknown token
            min_freq: Minimum frequency for a word to be included
            max_size: Maximum size of the vocabulary (excluding special tokens)
        """
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.min_freq = min_freq
        self.max_size = max_size
        
        self._word_counts: Counter = Counter()
    
    def add_words(self, words: Iterable[str]) -> None:
        """
        Add words to the vocabulary builder.
        
        Args:
            words: An iterable of words to add
        """
        self._word_counts.update(words)
    
    def add_sentences(self, sentences: Iterable[List[str]]) -> None:
        """
        Add multiple sentences (lists of words) to the vocabulary builder.
        
        Args:
            sentences: An iterable of sentences, each being a list of words
        """
        for sentence in sentences:
            self.add_words(sentence)
    
    def add_from_tokenized_texts(
        self, 
        tokenized_texts: Iterable[List[str]],
        tokenized_labels: Optional[Iterable[str]] = None
    ) -> None:
        """
        Add tokenized texts and optionally labels to the vocabulary builder.
        
        Args:
            tokenized_texts: An iterable of tokenized texts
            tokenized_labels: Optional iterable of labels
        """
        self.add_sentences(tokenized_texts)
        if tokenized_labels:
            self.add_words(tokenized_labels)
    
    def build(self) -> Vocabulary:
        """
        Build the vocabulary from the collected words.
        
        Returns:
            A Vocabulary instance
        """
        # Start with special tokens
        stoi: Dict[str, int] = {}
        idx = 0
        
        # Add special tokens first
        special_tokens = [self.pad_token, self.unk_token]
        for token in special_tokens:
            if token not in stoi:
                stoi[token] = idx
                idx += 1
        
        # Filter words by frequency and sort
        words_and_counts = [
            (word, count) 
            for word, count in self._word_counts.items()
            if count >= self.min_freq and word not in special_tokens
        ]
        
        # Sort by frequency (descending), then alphabetically
        words_and_counts.sort(key=lambda x: (-x[1], x[0]))
        
        # Limit by max_size if specified
        if self.max_size is not None:
            words_and_counts = words_and_counts[:self.max_size]
        
        # Add remaining words to vocabulary
        for word, _ in words_and_counts:
            stoi[word] = idx
            idx += 1
        
        return Vocabulary(stoi, self.pad_token, self.unk_token)
    
    def build_label_vocab(self, labels: Iterable[str]) -> Vocabulary:
        """
        Build a vocabulary specifically for labels.
        
        Args:
            labels: An iterable of labels
            
        Returns:
            A Vocabulary instance for labels
        """
        label_builder = VocabularyBuilder(
            pad_token=self.pad_token,
            unk_token=self.unk_token,
            min_freq=1
        )
        label_builder.add_words(labels)
        return label_builder.build()
