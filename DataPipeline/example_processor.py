# coding=utf-8
"""
Example Processor Module
This module provides classes for processing text examples into model-ready format.
"""

from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
from .vocabulary import Vocabulary, SpecialTokens
from .dataset_reader import TextCleaner


class ProcessedExample:
    """
    Container for a processed example.
    
    Attributes:
        text_indices: List of word indices for the text
        label_index: Index of the label
        text_length: Length of the text (before padding)
        raw_text: Original raw text (optional)
        raw_label: Original raw label (optional)
    """
    
    def __init__(
        self,
        text_indices: List[int],
        label_index: int,
        text_length: int,
        raw_text: Optional[str] = None,
        raw_label: Optional[str] = None
    ):
        self.text_indices = text_indices
        self.label_index = label_index
        self.text_length = text_length
        self.raw_text = raw_text
        self.raw_label = raw_label
    
    def __repr__(self) -> str:
        return (f"ProcessedExample(text_length={self.text_length}, "
                f"label_index={self.label_index})")


class ExampleProcessor(ABC):
    """
    Abstract base class for example processors.
    Responsible for converting raw (text, label) pairs into model-ready format.
    """
    
    @abstractmethod
    def process(self, text: str, label: str) -> ProcessedExample:
        """
        Process a single (text, label) pair.
        
        Args:
            text: Raw text string
            label: Raw label string
            
        Returns:
            A ProcessedExample instance
        """
        pass
    
    @abstractmethod
    def process_batch(
        self, 
        examples: List[Tuple[str, str]]
    ) -> List[ProcessedExample]:
        """
        Process a batch of (text, label) pairs.
        
        Args:
            examples: List of (text, label) tuples
            
        Returns:
            List of ProcessedExample instances
        """
        pass


class TextProcessor(ExampleProcessor):
    """
    Text processor for converting text examples.
    
    Handles:
    - Text cleaning
    - Tokenization
    - Word to index conversion
    - Label to index conversion
    """
    
    def __init__(
        self,
        text_vocab: Vocabulary,
        label_vocab: Vocabulary,
        lower: bool = True,
        clean_text: bool = True
    ):
        """
        Initialize the text processor.
        
        Args:
            text_vocab: Vocabulary for text tokens
            label_vocab: Vocabulary for labels
            lower: Whether to lowercase text
            clean_text: Whether to apply text cleaning
        """
        self.text_vocab = text_vocab
        self.label_vocab = label_vocab
        self.lower = lower
        self.clean_text = clean_text
        
        # Validate vocabularies
        self._validate_vocabs()
    
    def _validate_vocabs(self) -> None:
        """Validate that vocabularies are properly configured."""
        if self.text_vocab is None:
            raise ValueError("text_vocab cannot be None")
        if self.label_vocab is None:
            raise ValueError("label_vocab cannot be None")
    
    def _clean_and_tokenize(self, text: str) -> List[str]:
        """
        Clean and tokenize the input text.
        
        Args:
            text: Raw text string
            
        Returns:
            List of tokens
        """
        if self.clean_text:
            text = TextCleaner.clean_str(text)
        
        tokens = TextCleaner.tokenize(text, lower=self.lower)
        
        return tokens
    
    def process(self, text: str, label: str) -> ProcessedExample:
        """
        Process a single (text, label) pair.
        
        Args:
            text: Raw text string
            label: Raw label string
            
        Returns:
            A ProcessedExample instance
        """
        # Clean and tokenize text
        tokens = self._clean_and_tokenize(text)
        
        # Convert tokens to indices
        text_indices = self.text_vocab.lookup_words(tokens)
        
        # Convert label to index
        label_index = self.label_vocab.lookup_word(label)
        
        # Get text length
        text_length = len(text_indices)
        
        return ProcessedExample(
            text_indices=text_indices,
            label_index=label_index,
            text_length=text_length,
            raw_text=text,
            raw_label=label
        )
    
    def process_batch(
        self, 
        examples: List[Tuple[str, str]]
    ) -> List[ProcessedExample]:
        """
        Process a batch of (text, label) pairs.
        
        Args:
            examples: List of (text, label) tuples
            
        Returns:
            List of ProcessedExample instances
        """
        return [self.process(text, label) for text, label in examples]
    
    def process_without_label(self, text: str) -> ProcessedExample:
        """
        Process text without a label (for inference).
        
        Args:
            text: Raw text string
            
        Returns:
            A ProcessedExample instance with dummy label
        """
        tokens = self._clean_and_tokenize(text)
        text_indices = self.text_vocab.lookup_words(tokens)
        text_length = len(text_indices)
        
        return ProcessedExample(
            text_indices=text_indices,
            label_index=-1,  # Dummy label
            text_length=text_length,
            raw_text=text,
            raw_label=None
        )


class CharLevelProcessor(ExampleProcessor):
    """
    Character-level text processor.
    Converts text to character-level representations.
    """
    
    def __init__(
        self,
        char_vocab: Vocabulary,
        label_vocab: Vocabulary,
        lower: bool = True
    ):
        """
        Initialize the character-level processor.
        
        Args:
            char_vocab: Vocabulary for characters
            label_vocab: Vocabulary for labels
            lower: Whether to lowercase text
        """
        self.char_vocab = char_vocab
        self.label_vocab = label_vocab
        self.lower = lower
    
    def _char_tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into characters.
        
        Args:
            text: Raw text string
            
        Returns:
            List of characters
        """
        if self.lower:
            text = text.lower()
        
        # Split into characters, including spaces
        return list(text)
    
    def process(self, text: str, label: str) -> ProcessedExample:
        """
        Process a single (text, label) pair at character level.
        
        Args:
            text: Raw text string
            label: Raw label string
            
        Returns:
            A ProcessedExample instance
        """
        chars = self._char_tokenize(text)
        char_indices = self.char_vocab.lookup_words(chars)
        label_index = self.label_vocab.lookup_word(label)
        text_length = len(char_indices)
        
        return ProcessedExample(
            text_indices=char_indices,
            label_index=label_index,
            text_length=text_length,
            raw_text=text,
            raw_label=label
        )
    
    def process_batch(
        self, 
        examples: List[Tuple[str, str]]
    ) -> List[ProcessedExample]:
        """
        Process a batch of (text, label) pairs at character level.
        
        Args:
            examples: List of (text, label) tuples
            
        Returns:
            List of ProcessedExample instances
        """
        return [self.process(text, label) for text, label in examples]
