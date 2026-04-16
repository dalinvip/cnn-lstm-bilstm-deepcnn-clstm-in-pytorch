# coding=utf-8
"""
Dataset Reader Module
This module provides classes for reading and parsing raw data files.
"""

import os
import re
from abc import ABC, abstractmethod
from typing import List, Tuple, Iterator


class DatasetReader(ABC):
    """
    Abstract base class for dataset readers.
    Responsible for reading raw data files and extracting text and labels.
    """
    
    def __init__(self, file_path: str, shuffle: bool = True, seed: int = 233):
        """
        Initialize the dataset reader.
        
        Args:
            file_path: Path to the data file
            shuffle: Whether to shuffle the data after reading
            seed: Random seed for shuffling
        """
        self.file_path = file_path
        self.shuffle = shuffle
        self.seed = seed
        self._examples: List[Tuple[str, str]] = []
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
    
    @abstractmethod
    def _parse_line(self, line: str) -> Tuple[str, str]:
        """
        Parse a single line from the data file.
        
        Args:
            line: A line from the data file
            
        Returns:
            A tuple of (text, label)
        """
        pass
    
    def read(self) -> List[Tuple[str, str]]:
        """
        Read and parse the entire data file.
        
        Returns:
            A list of (text, label) tuples
        """
        self._examples = []
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                text, label = self._parse_line(line)
                if text and label:
                    self._examples.append((text, label))
        
        if self.shuffle:
            import random
            random.seed(self.seed)
            random.shuffle(self._examples)
        
        return self._examples
    
    def __iter__(self) -> Iterator[Tuple[str, str]]:
        """Iterate over the examples."""
        return iter(self._examples)
    
    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self._examples)
    
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """Get an example by index."""
        return self._examples[idx]


class TwoClassDatasetReader(DatasetReader):
    """
    Dataset reader for two-class classification tasks.
    Labels: 0,1 -> negative; 3,4 -> positive; 2 is neutral and skipped
    """
    
    def _parse_line(self, line: str) -> Tuple[str, str]:
        """
        Parse a line for two-class classification.
        
        Expected format: "text ||| label"
        Label mapping: 0,1 -> negative; 3,4 -> positive; 2 is skipped
        """
        try:
            # Split by ' ||| ' separator
            parts = line.split(' ||| ')
            if len(parts) != 2:
                return "", ""
            
            text = parts[0].strip()
            label_str = parts[1].strip()
            
            # Convert label
            if label_str in ['0', '1']:
                label = 'negative'
            elif label_str in ['3', '4']:
                label = 'positive'
            else:  # label 2 is neutral, skip
                return "", ""
            
            return text, label
            
        except Exception as e:
            return "", ""


class FiveClassDatasetReader(DatasetReader):
    """
    Dataset reader for five-class classification tasks.
    Labels: 0,1,2,3,4 are used as-is
    """
    
    def _parse_line(self, line: str) -> Tuple[str, str]:
        """
        Parse a line for five-class classification.
        
        Expected format: "text ||| label"
        Labels 0-4 are used directly
        """
        try:
            # Split by ' ||| ' separator
            parts = line.split(' ||| ')
            if len(parts) != 2:
                return "", ""
            
            text = parts[0].strip()
            label = parts[1].strip()
            
            # Validate label is 0-4
            if label not in ['0', '1', '2', '3', '4']:
                return "", ""
            
            return text, label
            
        except Exception as e:
            return "", ""


class TextCleaner:
    """
    Text cleaning utility for tokenization and string cleaning.
    """
    
    @staticmethod
    def clean_str(string: str) -> str:
        """
        Tokenization/string cleaning for datasets.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        
        Args:
            string: Input string to clean
            
        Returns:
            Cleaned string
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", r" ( ", string)
        string = re.sub(r"\)", r" ) ", string)
        string = re.sub(r"\?", r" ? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        
        return string.strip()
    
    @staticmethod
    def tokenize(string: str, lower: bool = True) -> List[str]:
        """
        Tokenize a string by splitting on whitespace.
        
        Args:
            string: Input string
            lower: Whether to lowercase the string
            
        Returns:
            List of tokens
        """
        if lower:
            string = string.lower()
        return string.split()
