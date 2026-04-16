# coding=utf-8
"""
Batch Collator Module
This module provides classes for assembling batches with padding.
"""

from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import torch
import numpy as np
from .example_processor import ProcessedExample
from .vocabulary import Vocabulary, SpecialTokens


class BatchData:
    """
    Container for a batch of data.
    
    Attributes:
        text_tensor: Tensor of text indices (batch_size, max_seq_length)
        label_tensor: Tensor of label indices (batch_size,)
        lengths: Tensor of sequence lengths (batch_size,)
        raw_examples: List of original ProcessedExample instances
    """
    
    def __init__(
        self,
        text_tensor: torch.Tensor,
        label_tensor: torch.Tensor,
        lengths: torch.Tensor,
        raw_examples: Optional[List[ProcessedExample]] = None
    ):
        self.text_tensor = text_tensor
        self.label_tensor = label_tensor
        self.lengths = lengths
        self.raw_examples = raw_examples
    
    @property
    def batch_size(self) -> int:
        """Return the batch size."""
        return self.text_tensor.size(0)
    
    @property
    def max_seq_length(self) -> int:
        """Return the maximum sequence length in the batch."""
        return self.text_tensor.size(1)
    
    def to(self, device: torch.device) -> 'BatchData':
        """
        Move all tensors to the specified device.
        
        Args:
            device: Target device
            
        Returns:
            A new BatchData instance with tensors on the specified device
        """
        return BatchData(
            text_tensor=self.text_tensor.to(device),
            label_tensor=self.label_tensor.to(device),
            lengths=self.lengths.to(device),
            raw_examples=self.raw_examples
        )
    
    def __repr__(self) -> str:
        return (f"BatchData(batch_size={self.batch_size}, "
                f"max_seq_length={self.max_seq_length})")


class BatchCollator(ABC):
    """
    Abstract base class for batch collators.
    Responsible for assembling multiple ProcessedExample instances into a batch.
    """
    
    @abstractmethod
    def collate(self, examples: List[ProcessedExample]) -> BatchData:
        """
        Collate a list of examples into a batch.
        
        Args:
            examples: List of ProcessedExample instances
            
        Returns:
            A BatchData instance
        """
        pass
    
    @abstractmethod
    def __call__(self, examples: List[ProcessedExample]) -> BatchData:
        """
        Make the collator callable.
        
        Args:
            examples: List of ProcessedExample instances
            
        Returns:
            A BatchData instance
        """
        pass


class PaddingCollator(BatchCollator):
    """
    Batch collator that pads sequences to the same length.
    
    Supports multiple padding strategies:
    - 'max': Pad to the maximum sequence length in the batch
    - 'fixed': Pad to a fixed length
    """
    
    PAD_STRATEGY_MAX = 'max'
    PAD_STRATEGY_FIXED = 'fixed'
    
    def __init__(
        self,
        pad_idx: int,
        pad_strategy: str = PAD_STRATEGY_MAX,
        max_length: Optional[int] = None,
        batch_first: bool = True,
        return_lengths: bool = True
    ):
        """
        Initialize the padding collator.
        
        Args:
            pad_idx: Index of the padding token
            pad_strategy: Padding strategy ('max' or 'fixed')
            max_length: Maximum sequence length (required for 'fixed' strategy)
            batch_first: Whether to return tensors with batch dimension first
            return_lengths: Whether to return sequence lengths
        """
        self.pad_idx = pad_idx
        self.pad_strategy = pad_strategy
        self.max_length = max_length
        self.batch_first = batch_first
        self.return_lengths = return_lengths
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate the collator configuration."""
        if self.pad_strategy not in [self.PAD_STRATEGY_MAX, self.PAD_STRATEGY_FIXED]:
            raise ValueError(f"Invalid pad_strategy: {self.pad_strategy}")
        
        if self.pad_strategy == self.PAD_STRATEGY_FIXED and self.max_length is None:
            raise ValueError("max_length must be specified for 'fixed' pad_strategy")
    
    def _get_target_length(self, examples: List[ProcessedExample]) -> int:
        """
        Determine the target length for padding.
        
        Args:
            examples: List of ProcessedExample instances
            
        Returns:
            The target sequence length
        """
        if self.pad_strategy == self.PAD_STRATEGY_MAX:
            max_len = max(ex.text_length for ex in examples)
            if self.max_length is not None:
                return min(max_len, self.max_length)
            return max_len
        else:  # fixed
            return self.max_length
    
    def _pad_sequence(
        self, 
        sequence: List[int], 
        target_length: int
    ) -> Tuple[List[int], int]:
        """
        Pad or truncate a sequence to the target length.
        
        Args:
            sequence: Input sequence
            target_length: Target length
            
        Returns:
            A tuple of (padded_sequence, actual_length)
        """
        actual_length = min(len(sequence), target_length)
        
        if len(sequence) >= target_length:
            # Truncate
            padded = sequence[:target_length]
        else:
            # Pad
            padded = sequence + [self.pad_idx] * (target_length - len(sequence))
        
        return padded, actual_length
    
    def collate(self, examples: List[ProcessedExample]) -> BatchData:
        """
        Collate a list of examples into a batch.
        
        Args:
            examples: List of ProcessedExample instances
            
        Returns:
            A BatchData instance
        """
        if not examples:
            raise ValueError("Cannot collate empty list of examples")
        
        # Determine target length
        target_length = self._get_target_length(examples)
        
        # Pad all sequences
        padded_sequences = []
        lengths = []
        labels = []
        
        for ex in examples:
            padded_seq, actual_len = self._pad_sequence(ex.text_indices, target_length)
            padded_sequences.append(padded_seq)
            lengths.append(actual_len)
            labels.append(ex.label_index)
        
        # Convert to tensors
        text_tensor = torch.tensor(padded_sequences, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        
        # Handle batch_first
        if not self.batch_first:
            text_tensor = text_tensor.transpose(0, 1)
        
        return BatchData(
            text_tensor=text_tensor,
            label_tensor=label_tensor,
            lengths=lengths_tensor,
            raw_examples=examples
        )
    
    def __call__(self, examples: List[ProcessedExample]) -> BatchData:
        """Make the collator callable."""
        return self.collate(examples)


class SortingPaddingCollator(PaddingCollator):
    """
    Batch collator that sorts sequences by length before padding.
    
    This is useful for RNNs that can take advantage of packed sequences.
    """
    
    def __init__(
        self,
        pad_idx: int,
        pad_strategy: str = PaddingCollator.PAD_STRATEGY_MAX,
        max_length: Optional[int] = None,
        batch_first: bool = True,
        return_lengths: bool = True,
        descending: bool = True
    ):
        """
        Initialize the sorting padding collator.
        
        Args:
            pad_idx: Index of the padding token
            pad_strategy: Padding strategy ('max' or 'fixed')
            max_length: Maximum sequence length
            batch_first: Whether to return tensors with batch dimension first
            return_lengths: Whether to return sequence lengths
            descending: Whether to sort in descending order
        """
        super().__init__(
            pad_idx=pad_idx,
            pad_strategy=pad_strategy,
            max_length=max_length,
            batch_first=batch_first,
            return_lengths=return_lengths
        )
        self.descending = descending
    
    def collate(self, examples: List[ProcessedExample]) -> BatchData:
        """
        Collate a list of examples into a batch, sorting by length first.
        
        Args:
            examples: List of ProcessedExample instances
            
        Returns:
            A BatchData instance with sorted sequences
        """
        if not examples:
            raise ValueError("Cannot collate empty list of examples")
        
        # Sort examples by length
        sorted_examples = sorted(
            examples,
            key=lambda x: x.text_length,
            reverse=self.descending
        )
        
        # Use parent class to pad
        return super().collate(sorted_examples)
