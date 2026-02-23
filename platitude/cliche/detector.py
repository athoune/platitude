"""
Cliche Detection System
Identifies common phrases in a text corpus using n-grams, BM25 scoring,
and optional embeddings for semantic analysis.
"""

from collections import Counter
import math
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# Optional: Uncomment for embeddings
# from sentence_transformers import SentenceTransformer


class ClicheDetector:
    """
    A class to detect clichés (common phrases) in a text corpus.

    Uses:
    - N-grams to capture multi-word expressions
    - BM25 scoring to measure phrase rarity
    - Optional embeddings for semantic similarity detection
    """

    def __init__(self, ngram_range=(2, 4), use_embeddings=False):
        """
        Initialize the detector.

        Args:
            ngram_range: Tuple of (min_n, max_n) for n-gram extraction
            use_embeddings: Whether to use sentence embeddings for semantic analysis
        """
        self.ngram_range = ngram_range
        self.use_embeddings = use_embeddings
        self.corpus = []
        self.ngrams_freq = Counter()
        self.bm25_scores = {}

        # Initialize embedding model if requested
        if use_embeddings:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except NameError:
                print("Warning: sentence_transformers not installed. Embeddings disabled.")
                self.use_embeddings = False

    def extract_ngrams(self, text: str, n: int) -> List[str]:
        """
        Extract n-grams from text.

        Args:
            text: Input text
            n: Size of n-gram

        Returns:
            List of n-grams
        """
        # Normalize: lowercase and split into words
        words = text.lower().split()

        # Create n-grams
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
        return ngrams

    def add_document(self, text: str):
        """
        Add a document to the corpus.

        Args:
            text: Document text
        """
        self.corpus.append(text)

        # Extract all n-grams in the specified range
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            ngrams = self.extract_ngrams(text, n)
            self.ngrams_freq.update(ngrams)

    def calculate_bm25_scores(self):
        """
        Calculate BM25 scores for all n-grams.

        BM25 is a ranking function that measures how "common" or "rare"
        a phrase is relative to its appearance pattern.

        Higher scores = rarer (more informative) phrases
        Lower scores = more common (clichéd) phrases
        """
        # Parameters for BM25
        k1 = 1.5  # Term frequency saturation parameter
        b = 0.75  # Length normalization parameter
        idf_scores = {}

        # Calculate IDF (Inverse Document Frequency) for each n-gram
        num_docs = len(self.corpus)

        for ngram in tqdm(self.ngrams_freq.keys(), desc="Ngam : Documents"):
            # Count documents containing this n-gram
            docs_with_ngram = sum(1 for doc in self.corpus if ngram in doc.lower())

            # IDF formula: log(total_docs / docs_containing_term)
            idf = math.log((num_docs - docs_with_ngram + 0.5) / (docs_with_ngram + 0.5) + 1)
            idf_scores[ngram] = idf

        # Calculate BM25 score for each n-gram
        avg_doc_length = sum(len(doc.split()) for doc in self.corpus) / num_docs

        for ngram in tqdm(self.ngrams_freq.keys(), desc="Ngram : BM25"):
            freq = self.ngrams_freq[ngram]
            idf = idf_scores[ngram]

            # BM25 formula
            bm25 = idf * ((k1 + 1) * freq) / (k1 * (1 - b + b * (len(ngram.split()) / avg_doc_length)) + freq)
            self.bm25_scores[ngram] = bm25

    def get_cliches(self, top_n=20, min_frequency=2) -> List[Tuple[str, int, float]]:
        """
        Get the most common clichés.

        Args:
            top_n: Number of clichés to return
            min_frequency: Minimum occurrence count to consider as cliché

        Returns:
            List of (ngram, frequency, rarity_score) tuples
        """
        # Calculate BM25 scores if not already done
        if not self.bm25_scores:
            self.calculate_bm25_scores()

        # Filter by minimum frequency
        cliches = [
            (ngram, self.ngrams_freq[ngram], self.bm25_scores[ngram])
            for ngram in self.ngrams_freq
            if self.ngrams_freq[ngram] >= min_frequency
        ]

        # Sort by rarity score (ascending) - lower scores = more clichéd
        cliches.sort(key=lambda x: x[2])

        return cliches[:top_n]

    def get_interesting_phrases(self, top_n=20, min_frequency=1) -> List[Tuple[str, int, float]]:
        """
        Get the most interesting (rare) phrases.
        Opposite of clichés - higher BM25 scores.

        Args:
            top_n: Number of phrases to return
            min_frequency: Minimum occurrence count

        Returns:
            List of (ngram, frequency, rarity_score) tuples
        """
        if not self.bm25_scores:
            self.calculate_bm25_scores()

        # Filter by minimum frequency
        phrases = [
            (ngram, self.ngrams_freq[ngram], self.bm25_scores[ngram])
            for ngram in self.ngrams_freq
            if self.ngrams_freq[ngram] >= min_frequency
        ]

        # Sort by rarity score (descending) - higher scores = more rare/interesting
        phrases.sort(key=lambda x: x[2], reverse=True)

        return phrases[:top_n]

    def find_cliche_instances(self, text: str) -> List[Tuple[str, int]]:
        """
        Find all clichés in a given text.

        Args:
            text: Text to analyze

        Returns:
            List of (cliche_phrase, bm25_score) tuples found in the text
        """
        if not self.bm25_scores:
            self.calculate_bm25_scores()

        cliche_threshold = np.percentile(list(self.bm25_scores.values()), 25)
        found_cliches = []

        for ngram, score in self.bm25_scores.items():
            if score < cliche_threshold and ngram in text.lower():
                found_cliches.append((ngram, score))

        # Sort by score (most clichéd first)
        found_cliches.sort(key=lambda x: x[1])
        return found_cliches

    def compare_semantic_similarity(self, phrases: List[str]) -> Dict:
        """
        Compare semantic similarity between phrases using embeddings.
        Useful to find clichés that are semantically similar (synonymous expressions).

        Args:
            phrases: List of phrases to compare

        Returns:
            Dictionary with similarity scores

        Note:
            This function is only available if use_embeddings=True
        """
        if not self.use_embeddings:
            print("Embeddings not enabled. Initialize with use_embeddings=True")
            return {}

        # Generate embeddings for each phrase
        embeddings = self.embedding_model.encode(phrases)

        # Compute cosine similarity matrix
        similarities = {}
        for i, phrase1 in enumerate(phrases):
            for j, phrase2 in enumerate(phrases):
                if i < j:
                    # Cosine similarity
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    key = f"{phrase1} <-> {phrase2}"
                    similarities[key] = float(similarity)

        return similarities


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Sample corpus
    corpus = [
        "It was a dark and stormy night when the hero arrived at the ancient castle.",
        "The dark and stormy night made it a perfect time for adventure in the castle.",
        "In this day and age, we must embrace new technologies with open arms.",
        "At the end of the day, it goes without saying that hard work pays off.",
        "The bottom line is that in this day and age, collaboration is key.",
        "It was a dark and stormy night, the kind that makes your heart race.",
        "In this day and age, success comes to those with open arms and determination.",
        "Last but not least, we should not forget the importance of hard work.",
        "At the end of the day, the bottom line remains unchanged despite everything.",
    ]

    # Initialize detector (without embeddings for this example)
    print("=== CLICHE DETECTION USING N-GRAMS AND BM25 ===\n")
    detector = ClicheDetector(ngram_range=(2, 4), use_embeddings=False)

    # Add all documents to corpus
    print("Processing corpus...")
    for doc in corpus:
        detector.add_document(doc)

    # Calculate BM25 scores
    print("Calculating BM25 scores...\n")
    detector.calculate_bm25_scores()

    # Display most common clichés
    print("TOP CLICHÉS (Lowest BM25 scores = Most common):")
    print("-" * 60)
    cliches = detector.get_cliches(top_n=10, min_frequency=2)
    for i, (phrase, freq, score) in enumerate(cliches, 1):
        print(f"{i:2d}. '{phrase}'")
        print(f"    Frequency: {freq}, BM25 Score: {score:.4f}\n")

    # Display most interesting phrases
    print("\nMOST INTERESTING PHRASES (Highest BM25 scores = Rarest):")
    print("-" * 60)
    interesting = detector.get_interesting_phrases(top_n=10, min_frequency=1)
    for i, (phrase, freq, score) in enumerate(interesting, 1):
        print(f"{i:2d}. '{phrase}'")
        print(f"    Frequency: {freq}, BM25 Score: {score:.4f}\n")

    # Find clichés in a new text
    print("\nFINDING CLICHÉS IN A NEW TEXT:")
    print("-" * 60)
    new_text = "In this day and age, at the end of the day, hard work pays off."
    found = detector.find_cliche_instances(new_text)
    print(f"Text: '{new_text}'")
    print(f"\nClichés found: {len(found)}")
    for phrase, score in found:
        print(f"  - '{phrase}' (BM25: {score:.4f})")

    # OPTIONAL: Test with embeddings
    print("\n\n=== SEMANTIC SIMILARITY (EMBEDDINGS OPTIONAL) ===")
    print("-" * 60)
    print("Note: To use embeddings, uncomment the import and set use_embeddings=True")
    print("This requires: pip install sentence-transformers")
    print("\nEmbeddings are useful for detecting semantically similar clichés,")
    print("such as: 'at the end of the day' vs 'when all is said and done'")

    # Uncomment below to test embeddings (requires sentence-transformers)
    # detector_with_embeddings = ClicheDetector(use_embeddings=True)
    # test_phrases = ["at the end of the day", "when all is said and done", "hard work pays off"]
    # similarities = detector_with_embeddings.compare_semantic_similarity(test_phrases)
    # for pair, sim_score in similarities.items():
    #     print(f"{pair}: {sim_score:.4f}")
