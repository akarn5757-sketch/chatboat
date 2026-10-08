"""
Text Preprocessor for FAQ Chatbot
Performs text cleaning, tokenization, stopword removal, and lemmatization using NLTK.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    """Preprocesses natural language text using NLTK tools."""

    def __init__(self, remove_stopwords: bool = True):
        self._ensure_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        self.remove_stopwords = remove_stopwords

        # Load standard English stopwords
        try:
            base_stopwords = set(stopwords.words("english"))
        except Exception:
            base_stopwords = {
                "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
                "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
                "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
                "their", "theirs", "themselves", "a", "an", "the", "and", "but", "if", "or",
                "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
                "against", "between", "into", "through", "during", "before", "after", "above",
                "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
                "again", "further", "then", "once", "here", "there", "all", "any", "both", "each",
                "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
                "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should"
            }

        # Use standard stopwords
        self.stop_words = base_stopwords

    @staticmethod
    def _ensure_nltk_resources():
        """Ensures all required NLTK corpora are downloaded."""
        resources = [
            ("tokenizers/punkt", "punkt"),
            ("tokenizers/punkt_tab", "punkt_tab"),
            ("corpora/stopwords", "stopwords"),
            ("corpora/wordnet", "wordnet"),
            ("corpora/omw-1.4", "omw-1.4"),
        ]
        for path, name in resources:
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(name, quiet=True)
                except Exception:
                    pass

    def clean_text(self, text: str) -> str:
        """Lowercases text, removes special characters and excess whitespace."""
        if not text or not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()

        # Expand common contractions
        contractions = {
            r"can\'t": "cannot",
            r"won\'t": "will not",
            r"n\'t": " not",
            r"\'re": " are",
            r"\'s": " is",
            r"\'d": " would",
            r"\'ll": " will",
            r"\'t": " not",
            r"\'ve": " have",
            r"\'m": " am",
        }
        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text)

        # Remove special characters, keep letters and digits
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> list[str]:
        """Tokenizes text into words with fallback to regex if punkt fails."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return []
        try:
            tokens = word_tokenize(cleaned)
        except Exception:
            tokens = re.findall(r"\b\w+\b", cleaned)
        return tokens

    def preprocess_tokens(self, text: str) -> list[str]:
        """Tokenizes, filters stopwords, and lemmatizes tokens."""
        tokens = self.tokenize(text)
        processed = []
        for token in tokens:
            # Skip pure punctuation
            if token in string.punctuation:
                continue

            # Optional stopword filter
            if self.remove_stopwords and token in self.stop_words:
                continue

            # Lemmatize (try verb then noun for better root word matching)
            lemmatized = self.lemmatizer.lemmatize(token, pos="v")
            if lemmatized == token:
                lemmatized = self.lemmatizer.lemmatize(token, pos="n")

            if len(lemmatized) > 1 or lemmatized.isalnum():
                processed.append(lemmatized)

        return processed

    def preprocess(self, text: str) -> str:
        """Returns preprocessed text as a single normalized string."""
        tokens = self.preprocess_tokens(text)
        return " ".join(tokens)
