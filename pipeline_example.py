# coding=utf-8
"""
Example script demonstrating the usage of the refactored DataPipeline module.
This script shows how to use the modular data processing pipeline.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional

# Import from the refactored DataPipeline module
from DataPipeline import (
    TwoClassDatasetReader,
    FiveClassDatasetReader,
    VocabularyBuilder,
    TextProcessor,
    PaddingCollator,
    SortingPaddingCollator,
    EmbeddingInitializer,
    ProcessedExample
)


class TextClassificationDataset(Dataset):
    """
    PyTorch Dataset wrapper for text classification.
    Uses the refactored DataPipeline components.
    """
    
    def __init__(
        self,
        examples: List[ProcessedExample]
    ):
        """
        Initialize the dataset.
        
        Args:
            examples: List of ProcessedExample instances
        """
        self.examples = examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> ProcessedExample:
        return self.examples[idx]


def run_two_classification_example():
    """
    Example: Two-class classification using the refactored pipeline.
    """
    print("=" * 60)
    print("Example 1: Two-class Classification Pipeline")
    print("=" * 60)
    
    # Configuration
    train_file = "./Data/raw.clean.train"
    dev_file = "./Data/raw.clean.dev"
    test_file = "./Data/raw.clean.test"
    word_embedding_file = "./word2vec/glove.sentiment.conj.pretrained.txt"
    
    min_freq = 1
    batch_size = 16
    seed = 233
    
    # Step 1: Read raw data using DatasetReader
    print("\n[Step 1] Reading raw data...")
    
    train_reader = TwoClassDatasetReader(train_file, shuffle=True, seed=seed)
    dev_reader = TwoClassDatasetReader(dev_file, shuffle=False, seed=seed)
    test_reader = TwoClassDatasetReader(test_file, shuffle=False, seed=seed)
    
    train_examples_raw = train_reader.read()
    dev_examples_raw = dev_reader.read()
    test_examples_raw = test_reader.read()
    
    print(f"  Train examples: {len(train_examples_raw)}")
    print(f"  Dev examples: {len(dev_examples_raw)}")
    print(f"  Test examples: {len(test_examples_raw)}")
    
    # Step 2: Build vocabulary using VocabularyBuilder
    print("\n[Step 2] Building vocabulary...")
    
    # First, we need to tokenize all texts to build vocabulary
    # We'll use the TextCleaner from dataset_reader
    from DataPipeline.dataset_reader import TextCleaner
    
    # Tokenize all texts for vocabulary building
    all_tokenized_texts = []
    all_labels = []
    
    for text, label in train_examples_raw:
        cleaned = TextCleaner.clean_str(text)
        tokens = TextCleaner.tokenize(cleaned, lower=True)
        all_tokenized_texts.append(tokens)
        all_labels.append(label)
    
    # Build text vocabulary
    text_vocab_builder = VocabularyBuilder(
        min_freq=min_freq,
        pad_token="<pad>",
        unk_token="<unk>"
    )
    text_vocab_builder.add_sentences(all_tokenized_texts)
    text_vocab = text_vocab_builder.build()
    
    # Build label vocabulary
    label_vocab_builder = VocabularyBuilder(
        min_freq=1,
        pad_token="<pad>",
        unk_token="<unk>"
    )
    label_vocab = label_vocab_builder.build_label_vocab(all_labels)
    
    print(f"  Text vocabulary size: {len(text_vocab)}")
    print(f"  Label vocabulary size: {len(label_vocab)}")
    print(f"  Label mapping: {label_vocab.stoi}")
    print(f"  Padding index: {text_vocab.pad_idx}")
    print(f"  Unknown index: {text_vocab.unk_idx}")
    
    # Step 3: Process examples using ExampleProcessor
    print("\n[Step 3] Processing examples...")
    
    text_processor = TextProcessor(
        text_vocab=text_vocab,
        label_vocab=label_vocab,
        lower=True,
        clean_text=True
    )
    
    # Process all splits
    train_processed = text_processor.process_batch(train_examples_raw)
    dev_processed = text_processor.process_batch(dev_examples_raw)
    test_processed = text_processor.process_batch(test_examples_raw)
    
    # Show some processing examples
    print(f"  Processed train examples: {len(train_processed)}")
    print(f"  Example processed sample:")
    if train_processed:
        sample = train_processed[0]
        print(f"    Raw text: {sample.raw_text[:100]}...")
        print(f"    Raw label: {sample.raw_label}")
        print(f"    Text indices: {sample.text_indices[:10]}...")
        print(f"    Label index: {sample.label_index}")
        print(f"    Text length: {sample.text_length}")
    
    # Step 4: Create PyTorch Datasets and DataLoaders
    print("\n[Step 4] Creating DataLoaders...")
    
    train_dataset = TextClassificationDataset(train_processed)
    dev_dataset = TextClassificationDataset(dev_processed)
    test_dataset = TextClassificationDataset(test_processed)
    
    # Create batch collator
    collator = PaddingCollator(
        pad_idx=text_vocab.pad_idx,
        pad_strategy='max',
        batch_first=True
    )
    
    # For RNNs, you might want to use SortingPaddingCollator
    # collator = SortingPaddingCollator(
    #     pad_idx=text_vocab.pad_idx,
    #     pad_strategy='max',
    #     batch_first=True,
    #     descending=True
    # )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Dev batches: {len(dev_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Show a batch example
    print("\n  Sample batch:")
    for batch in train_loader:
        print(f"    Text tensor shape: {batch.text_tensor.shape}")
        print(f"    Label tensor shape: {batch.label_tensor.shape}")
        print(f"    Lengths tensor shape: {batch.lengths.shape}")
        print(f"    Batch size: {batch.batch_size}")
        print(f"    Max sequence length: {batch.max_seq_length}")
        break
    
    # Step 5: Load pretrained embeddings
    print("\n[Step 5] Loading pretrained embeddings...")
    
    if os.path.exists(word_embedding_file):
        # Create embedding loader with average strategy for OOV words
        embed_loader = EmbeddingInitializer.create_loader(
            file_path=word_embedding_file,
            strategy='average',  # Options: 'zero', 'uniform', 'average', 'xavier'
            verbose=True
        )
        
        # Load embeddings for our vocabulary
        embed_result = embed_loader.load_for_vocab(text_vocab)
        
        print(f"\n  Embedding loaded successfully:")
        print(f"    Vocab size: {embed_result.vocab_size}")
        print(f"    Embedding dim: {embed_result.embedding_dim}")
        print(f"    IV words: {embed_result.iv_count}")
        print(f"    OOV words: {embed_result.oov_count}")
        print(f"    OOV ratio: {embed_result.oov_ratio:.4f}")
        
        # The embedding matrix can be used to initialize nn.Embedding
        # embedding = nn.Embedding.from_pretrained(embed_result.embedding_matrix, freeze=False)
    else:
        print(f"  Warning: Pretrained embedding file not found: {word_embedding_file}")
        print("  Skipping pretrained embedding loading.")
    
    print("\n" + "=" * 60)
    print("Example 1 completed successfully!")
    print("=" * 60)
    
    # Return useful objects for potential further use
    return {
        'text_vocab': text_vocab,
        'label_vocab': label_vocab,
        'train_loader': train_loader,
        'dev_loader': dev_loader,
        'test_loader': test_loader,
        'text_processor': text_processor,
        'collator': collator
    }


def run_inference_example(text_processor: TextProcessor, collator: PaddingCollator):
    """
    Example: Inference using the trained pipeline.
    
    Args:
        text_processor: The text processor used during training
        collator: The batch collator used during training
    """
    print("\n" + "=" * 60)
    print("Example 2: Inference Pipeline")
    print("=" * 60)
    
    # Sample texts for inference
    sample_texts = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "Terrible acting, poor direction, and a confusing plot. Waste of time.",
        "The cinematography was beautiful, but the story was a bit slow."
    ]
    
    print("\n[Step 1] Processing sample texts for inference...")
    
    # Process each text (without labels)
    processed_examples = []
    for i, text in enumerate(sample_texts):
        processed = text_processor.process_without_label(text)
        processed_examples.append(processed)
        
        print(f"\n  Text {i+1}: {text[:80]}...")
        print(f"    Token indices: {processed.text_indices[:10]}...")
        print(f"    Length: {processed.text_length}")
    
    # Step 2: Create a batch for inference
    print("\n[Step 2] Creating inference batch...")
    
    batch = collator(processed_examples)
    
    print(f"  Batch text shape: {batch.text_tensor.shape}")
    print(f"  Batch lengths: {batch.lengths}")
    
    print("\n" + "=" * 60)
    print("Example 2 completed successfully!")
    print("=" * 60)


def run_five_classification_example():
    """
    Example: Five-class classification using the refactored pipeline.
    """
    print("\n" + "=" * 60)
    print("Example 3: Five-class Classification Pipeline")
    print("=" * 60)
    
    # Configuration
    train_file = "./Data/raw.clean.train"
    
    # Step 1: Read raw data using FiveClassDatasetReader
    print("\n[Step 1] Reading raw data for 5-class classification...")
    
    train_reader = FiveClassDatasetReader(train_file, shuffle=True, seed=233)
    train_examples_raw = train_reader.read()
    
    print(f"  Train examples: {len(train_examples_raw)}")
    
    # Step 2: Build vocabulary
    print("\n[Step 2] Building vocabulary...")
    
    from DataPipeline.dataset_reader import TextCleaner
    
    all_tokenized_texts = []
    all_labels = []
    
    for text, label in train_examples_raw:
        cleaned = TextCleaner.clean_str(text)
        tokens = TextCleaner.tokenize(cleaned, lower=True)
        all_tokenized_texts.append(tokens)
        all_labels.append(label)
    
    # Build vocabularies
    text_vocab_builder = VocabularyBuilder(min_freq=1)
    text_vocab_builder.add_sentences(all_tokenized_texts)
    text_vocab = text_vocab_builder.build()
    
    label_vocab_builder = VocabularyBuilder(min_freq=1)
    label_vocab = label_vocab_builder.build_label_vocab(all_labels)
    
    print(f"  Text vocabulary size: {len(text_vocab)}")
    print(f"  Label vocabulary size: {len(label_vocab)}")
    print(f"  Label mapping (5 classes): {label_vocab.stoi}")
    
    # Step 3: Process examples
    print("\n[Step 3] Processing examples...")
    
    text_processor = TextProcessor(
        text_vocab=text_vocab,
        label_vocab=label_vocab,
        lower=True,
        clean_text=True
    )
    
    train_processed = text_processor.process_batch(train_examples_raw)
    
    print(f"  Processed train examples: {len(train_processed)}")
    
    # Step 4: Create DataLoader with fixed length padding
    print("\n[Step 4] Creating DataLoader with fixed-length padding...")
    
    train_dataset = TextClassificationDataset(train_processed)
    
    # Use fixed-length padding (e.g., for CNNs that require fixed input size)
    collator = PaddingCollator(
        pad_idx=text_vocab.pad_idx,
        pad_strategy='fixed',
        max_length=100,  # Fixed sequence length
        batch_first=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collator
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Using fixed sequence length: 100")
    
    # Show a batch example
    print("\n  Sample batch with fixed length:")
    for batch in train_loader:
        print(f"    Text tensor shape: {batch.text_tensor.shape}")
        print(f"    Label tensor shape: {batch.label_tensor.shape}")
        break
    
    print("\n" + "=" * 60)
    print("Example 3 completed successfully!")
    print("=" * 60)


def main():
    """
    Main function to run all examples.
    """
    print("\n" + "=" * 60)
    print("DataPipeline Refactoring - Complete Usage Examples")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(233)
    
    # Run Example 1: Two-class classification
    results = run_two_classification_example()
    
    # Run Example 2: Inference using the trained pipeline
    run_inference_example(results['text_processor'], results['collator'])
    
    # Run Example 3: Five-class classification
    run_five_classification_example()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
