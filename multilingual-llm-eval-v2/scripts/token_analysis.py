import re
import argparse
from collections import Counter
from typing import List, Set, Tuple, Iterator
import os
import csv
import datetime # Import datetime for timestamp

# CORRECTED IMPORTS:
from tokenizers import Tokenizer # Tokenizer is now the main class
from tokenizers import models    # Contains BPE, WordPiece, etc.
from tokenizers import pre_tokenizers # Contains Whitespace, etc.
from tokenizers import trainers  # Contains BpeTrainer, WordPieceTrainer, etc.


# --- 1. Custom Tokenizer Wrapper for Hugging Face Tokenizers ---
class HfCustomTokenizer:
    def __init__(self, tokenizer: Tokenizer, unk_token: str = "<unk>"):
        self.tokenizer = tokenizer
        self.unk_token = unk_token
        self._vocab = set(self.tokenizer.get_vocab().keys())

    def encode(self, text: str) -> List[str]:
        return self.tokenizer.encode(text).tokens

    def get_vocab(self) -> Set[str]:
        return self._vocab

# --- 2. Training a Custom Tokenizer with Hugging Face Tokenizers ---
def train_custom_subword_tokenizer(
    corpus_iterator: Iterator[str],
    vocab_size: int = 30000,
    min_frequency: int = 5,
    special_tokens: List[str] = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]
) -> HfCustomTokenizer:
    """
    Trains a Byte Pair Encoding (BPE) tokenizer using the `tokenizers` library.
    """
    print(f"Training tokenizer with vocab_size={vocab_size} and min_frequency={min_frequency}...")
    
    # CORRECTED: Use models.BPE() instead of directly BytePairEncoding
    tokenizer = Tokenizer(models.BPE()) 
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # CORRECTED: Use trainers.BpeTrainer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens
    )

    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer, length=1_000_000) 
    
    print(f"Tokenizer training complete. Vocabulary size: {len(tokenizer.get_vocab())}")
    return HfCustomTokenizer(tokenizer)


# --- 3. Preprocessing and Analysis Functions ---

def simple_word_tokenize(text: str) -> List[str]:
    """
    Performs a very basic word tokenization using whitespace and punctuation.
    This serves as our 'ground truth' for words.
    """
    words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
    return [word for word in words if word.strip()]

def analyze_tokenization(
    texts: Iterator[str],
    custom_tokenizer: HfCustomTokenizer,
    word_level_tokenizer_fn: callable,
    sample_limit: int = 10000
) -> Tuple[float, float, float, float, int, int]: # Added tokenizer_vocab_size and processed_texts_count
    """
    Analyzes token coverage metrics across multiple texts.
    Returns: (avg_tokens_per_word, avg_oov_rate, avg_vocab_coverage, avg_subword_fragmentation_rate,
              actual_tokenizer_vocab_size, num_processed_texts)
    """
    total_words = 0
    total_custom_tokens = 0
    total_unique_words_in_sample = set()
    total_fragmented_words_count = 0
    num_processed_texts = 0

    custom_vocab = custom_tokenizer.get_vocab()
    tokenizer_vocab_size = len(custom_vocab) # Get the actual vocab size

    print(f"\nAnalyzing tokenization for up to {sample_limit} texts...")

    for text in texts:
        if num_processed_texts >= sample_limit:
            break
        
        words = word_level_tokenizer_fn(text)
        custom_tokens = custom_tokenizer.encode(text)

        total_words += len(words)
        total_custom_tokens += len(custom_tokens)
        total_unique_words_in_sample.update(words)

        for word in words:
            word_tokens = custom_tokenizer.encode(word)
            if len(word_tokens) > 1:
                total_fragmented_words_count += 1
        
        num_processed_texts += 1
        if num_processed_texts % (sample_limit // 10 if sample_limit > 10 else 1) == 0:
            print(f"  Processed {num_processed_texts} texts...")
            
    if num_processed_texts == 0:
        print("Warning: No texts were processed for analysis. Ensure your dataset has text data.")
        return 0.0, 0.0, 0.0, 0.0, tokenizer_vocab_size, 0

    if total_words == 0:
        print("Warning: No words found in the processed texts for analysis.")
        return 0.0, 0.0, 0.0, 0.0, tokenizer_vocab_size, num_processed_texts

    avg_tokens_per_word = total_custom_tokens / total_words

    oov_words_count = 0
    in_vocabulary_words_count = 0
    for word in total_unique_words_in_sample:
        if word not in custom_vocab:
            oov_words_count += 1
        else:
            in_vocabulary_words_count += 1

    avg_oov_rate = oov_words_count / len(total_unique_words_in_sample) if len(total_unique_words_in_sample) > 0 else 0
    avg_vocab_coverage = in_vocabulary_words_count / len(total_unique_words_in_sample) if len(total_unique_words_in_sample) > 0 else 0
    avg_subword_fragmentation_rate = total_fragmented_words_count / total_words

    return avg_tokens_per_word, avg_oov_rate, avg_vocab_coverage, avg_subword_fragmentation_rate, \
           tokenizer_vocab_size, num_processed_texts

# --- Main Execution with Argparse ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze token coverage on Hugging Face datasets for low-resource languages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="uonlp/CulturaX",
        help="Hugging Face dataset name (e.g., 'uonlp/CulturaX', 'oscar', 'mc4')."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="fi",
        help="Language code for the dataset (e.g., 'fi', 'sw', 'ar'). For CulturaX, check its page."
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=500000,
        help="Number of samples from the dataset to use for tokenizer training."
    )
    parser.add_argument(
        "--analysis_samples",
        type=int,
        default=50000,
        help="Number of samples from the dataset to use for metric analysis."
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=30000,
        help="Target vocabulary size for the custom tokenizer."
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=5,
        help="Minimum frequency for a token to be included in the tokenizer vocabulary."
    )
    parser.add_argument(
        "--output_csv_file",
        type=str,
        default="token_analysis_results.csv",
        help="Path to a CSV file to save the results. Results will be appended."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="token_analysis_reports",
        help="Directory to save the output CSV file."
    )

    args = parser.parse_args()

    # Determine output CSV path
    output_csv_path = args.output_csv_file
    if not os.path.isabs(output_csv_path):
        os.makedirs(args.output_dir, exist_ok=True)
        output_csv_path = os.path.join(args.output_dir, output_csv_path)

    # Check if the CSV file already exists to decide whether to write headers
    file_exists = os.path.exists(output_csv_path)

    # Define the CSV header
    csv_header = [
        "timestamp",
        "dataset",
        "language",
        "train_samples",
        "analysis_samples",
        "target_vocab_size",
        "min_frequency",
        "actual_tokenizer_vocab_size",
        "processed_analysis_texts",
        "avg_tokens_per_word",
        "avg_oov_rate",
        "avg_vocab_coverage",
        "avg_subword_fragmentation_rate"
    ]

    # Open CSV file in append mode
    csv_file = open(output_csv_path, 'a', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)

    # Write header only if file didn't exist
    if not file_exists:
        csv_writer.writerow(csv_header)
        print(f"Created new CSV file: {output_csv_path}")
    else:
        print(f"Appending results to existing CSV file: {output_csv_path}")

    print("--- Starting Token Coverage Analysis ---")
    print(f"Dataset: {args.dataset}")
    print(f"Language: {args.language}")
    print(f"Tokenizer Training Samples: {args.train_samples}")
    print(f"Analysis Samples: {args.analysis_samples}")
    print(f"Tokenizer Target Vocab Size: {args.vocab_size}")
    print(f"Tokenizer Min Frequency: {args.min_frequency}")

    try:
        print(f"\nLoading dataset '{args.dataset}' for language: '{args.language}'...")
        dataset = load_dataset(args.dataset, args.language, split="train", streaming=True)
        print("Dataset loaded in streaming mode.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have accepted the terms on the Hugging Face page if prompted.")
        print("You might need to run `huggingface-cli login` in your terminal and accept the terms.")
        csv_file.close()
        exit()

    # --- Prepare data for tokenizer training ---
    print(f"\nPreparing {args.train_samples} samples for tokenizer training...")
    # It's crucial to create a new iterator each time you want to iterate from the beginning
    # of a streaming dataset. `.take()` and then converting to a list or similar could also work
    # if you have enough memory for the training samples. For very large datasets,
    # passing the iterator directly to `train_from_iterator` as done below is efficient.
    
    # Reload dataset for training samples to ensure a fresh iterator
    # This is important because the 'dataset' object might be exhausted if iterated fully previously.
    train_dataset_for_iter = load_dataset(args.dataset, args.language, split="train", streaming=True)
    tokenizer_training_texts = (example["text"] for i, example in enumerate(train_dataset_for_iter) if i < args.train_samples)


    # --- Train the custom tokenizer ---
    custom_tokenizer = train_custom_subword_tokenizer(
        tokenizer_training_texts,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency
    )

    # Reset dataset iterator for analysis
    print(f"\nPreparing {args.analysis_samples} samples for metric analysis...")
    # Reload dataset for analysis samples to ensure a fresh iterator for analysis
    analysis_dataset_for_iter = load_dataset(args.dataset, args.language, split="train", streaming=True)
    analysis_texts = (example["text"] for i, example in enumerate(analysis_dataset_for_iter) if i < args.analysis_samples)

    # --- Perform the analysis ---
    avg_tpw, avg_oov_rate, avg_vocab_coverage, avg_fragmentation_rate, \
    actual_tokenizer_vocab_size, processed_analysis_texts = analyze_tokenization(
        analysis_texts,
        custom_tokenizer,
        simple_word_tokenize,
        sample_limit=args.analysis_samples
    )

    print("\n--- Final Token Coverage Metrics ---")
    print(f"Language: {args.language}")
    print(f"Actual Tokenizer Vocabulary Size: {actual_tokenizer_vocab_size}")
    print(f"Processed Analysis Texts: {processed_analysis_texts}")
    print(f"Average Tokens per Word (TPW): {avg_tpw:.4f}")
    print(f"Average Out-of-Vocabulary (OOV) Rate: {avg_oov_rate:.4f}")
    print(f"Average Vocabulary Coverage: {avg_vocab_coverage:.4f}")
    print(f"Average Subword Fragmentation Rate: {avg_fragmentation_rate:.4f}")

    # Prepare data row for CSV
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    csv_data_row = [
        timestamp,
        args.dataset,
        args.language,
        args.train_samples,
        args.analysis_samples,
        args.vocab_size,
        args.min_frequency,
        actual_tokenizer_vocab_size,
        processed_analysis_texts,
        f"{avg_tpw:.4f}",
        f"{avg_oov_rate:.4f}",
        f"{avg_vocab_coverage:.4f}",
        f"{avg_fragmentation_rate:.4f}"
    ]

    csv_writer.writerow(csv_data_row)
    csv_file.close()

    print(f"\nAnalysis complete. Results saved to: {output_csv_path}")