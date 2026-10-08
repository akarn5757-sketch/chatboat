"""
FAQ Engine for CodeAlpha Chatbot
Performs TF-IDF Vectorization and Cosine Similarity Matching to find
the most relevant FAQ answer for any user inquiry.
"""

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessor import TextPreprocessor


class FAQChatbot:
    """Conversational FAQ Chatbot driven by NLP preprocessing,

    TF-IDF Vectorization, and Cosine Similarity.
    """

    DEFAULT_GREETINGS = [
        "hello", "hi", "hey", "greetings", "good morning", "good afternoon",
        "good evening", "howdy", "sup", "yo"
    ]
    GREETING_RESPONSES = [
        "Hello! 👋 How can I help you today? Feel free to ask about our orders, shipping, payments, or returns.",
        "Hi there! Welcome to Customer Support. What can I assist you with today?",
        "Hey! I'm your FAQ assistant. Ask me anything about our products, delivery, or account services!",
    ]

    FAREWELL_INPUTS = ["bye", "goodbye", "see you", "exit", "quit", "cya", "farewell"]
    FAREWELL_RESPONSES = [
        "Goodbye! Have a wonderful day ahead! 😊",
        "Thanks for stopping by! If you need anything else, feel free to ask anytime.",
        "Take care! Don't hesitate to reach out if you have more questions.",
    ]

    GRATITUDE_INPUTS = ["thank you", "thanks", "thx", "appreciate it", "much appreciated"]
    GRATITUDE_RESPONSES = [
        "You're very welcome! Let me know if you need anything else. 😊",
        "Glad I could help! Is there anything else I can answer for you?",
        "Happy to assist! Feel free to ask more questions anytime.",
    ]

    def __init__(
        self,
        faq_filepath: Optional[str] = None,
        similarity_threshold: float = 0.25,
    ):
        """Initializes the FAQ chatbot.

        :param faq_filepath: Path to the JSON file containing FAQ data.
        :param similarity_threshold: Minimum cosine similarity required for a match.
        """
        self.preprocessor = TextPreprocessor(remove_stopwords=True)
        self.similarity_threshold = similarity_threshold

        # Default path to data/faqs.json
        if faq_filepath is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            faq_filepath = os.path.join(base_dir, "data", "faqs.json")

        self.faq_filepath = faq_filepath
        self.raw_faqs: List[Dict[str, Any]] = []
        self.indexed_items: List[Dict[str, Any]] = []
        self.corpus_preprocessed: List[str] = []

        # Scikit-learn TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.tfidf_matrix = None

        self.load_faqs(self.faq_filepath)

    def load_faqs(self, filepath: str) -> None:
        """Loads and indexes FAQs from JSON file, building the TF-IDF matrix."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"FAQ file not found at: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            self.raw_faqs = json.load(f)

        self.indexed_items = []
        self.corpus_preprocessed = []

        # Index primary question and all alternative patterns for high recall
        for faq in self.raw_faqs:
            faq_id = faq.get("id")
            category = faq.get("category", "General")
            main_question = faq.get("question", "")
            answer = faq.get("answer", "")
            patterns = faq.get("patterns", [])

            all_questions = [main_question] + patterns
            for q in all_questions:
                if not q.strip():
                    continue
                preprocessed_q = self.preprocessor.preprocess(q)
                self.indexed_items.append({
                    "faq_id": faq_id,
                    "category": category,
                    "pattern": q,
                    "primary_question": main_question,
                    "answer": answer,
                    "preprocessed": preprocessed_q,
                })
                self.corpus_preprocessed.append(preprocessed_q)

        # Fit TF-IDF on corpus
        if self.corpus_preprocessed:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus_preprocessed)

    def _check_conversational_intents(self, user_text: str) -> Optional[Dict[str, Any]]:
        """Handles chit-chat, greetings, gratitude, and exit intents."""
        cleaned = user_text.lower().strip()
        cleaned_no_punct = "".join(c for c in cleaned if c.isalnum() or c.isspace()).strip()

        # Check greetings
        if any(cleaned_no_punct == g or cleaned_no_punct.startswith(g + " ") for g in self.DEFAULT_GREETINGS):
            return {
                "answer": random.choice(self.GREETING_RESPONSES),
                "confidence": 1.0,
                "matched_question": "Greeting",
                "category": "Small Talk",
                "intent": "greeting",
                "suggestions": self.get_sample_questions(3),
            }

        # Check gratitude
        if any(g in cleaned_no_punct for g in self.GRATITUDE_INPUTS):
            return {
                "answer": random.choice(self.GRATITUDE_RESPONSES),
                "confidence": 1.0,
                "matched_question": "Thank You",
                "category": "Small Talk",
                "intent": "gratitude",
                "suggestions": [],
            }

        # Check farewell
        if any(cleaned_no_punct == f or cleaned_no_punct.startswith(f + " ") for f in self.FAREWELL_INPUTS):
            return {
                "answer": random.choice(self.FAREWELL_RESPONSES),
                "confidence": 1.0,
                "matched_question": "Farewell",
                "category": "Small Talk",
                "intent": "farewell",
                "suggestions": [],
            }

        # Check bot identity / help
        if any(q in cleaned_no_punct for q in ["who are you", "what can you do", "help", "what do you do"]):
            return {
                "answer": (
                    "I am an intelligent FAQ Assistant built with Python, NLTK, and Scikit-learn. "
                    "I can answer questions regarding orders, shipping, delivery tracking, "
                    "returns, refunds, payments, security, and warranties."
                ),
                "confidence": 1.0,
                "matched_question": "Bot Help & Capabilities",
                "category": "Help",
                "intent": "help",
                "suggestions": self.get_sample_questions(4),
            }

        return None

    def match_question(self, user_query: str) -> Tuple[Optional[Dict[str, Any]], float, List[Dict[str, Any]]]:
        """Calculates cosine similarity between user query and all FAQ patterns.

        Returns (best_match_item, max_score, top_candidates).
        """
        if not user_query or not user_query.strip():
            return None, 0.0, []

        preprocessed_query = self.preprocessor.preprocess(user_query)

        # If after cleaning we have no tokens, return no match
        if not preprocessed_query.strip():
            return None, 0.0, []

        # Vectorize query
        query_vector = self.vectorizer.transform([preprocessed_query])

        # Compute cosine similarity
        cosine_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        if len(cosine_scores) == 0:
            return None, 0.0, []

        best_idx = int(np.argmax(cosine_scores))
        max_score = float(cosine_scores[best_idx])

        # Find top 3 distinct FAQ recommendations
        top_indices = np.argsort(cosine_scores)[::-1]
        top_candidates = []
        seen_faq_ids = set()

        for idx in top_indices:
            score = float(cosine_scores[idx])
            item = self.indexed_items[idx]
            faq_id = item["faq_id"]

            if faq_id not in seen_faq_ids and score > 0.10:
                seen_faq_ids.add(faq_id)
                top_candidates.append({
                    "question": item["primary_question"],
                    "category": item["category"],
                    "score": round(score, 3),
                    "answer": item["answer"],
                })
            if len(top_candidates) >= 3:
                break

        best_item = self.indexed_items[best_idx] if max_score > 0.0 else None
        return best_item, max_score, top_candidates

    def get_response(self, user_query: str) -> Dict[str, Any]:
        """Main method to process a user query and return a formatted chatbot response."""
        user_query_clean = user_query.strip() if user_query else ""

        if not user_query_clean:
            return {
                "answer": "Please ask a question so I can assist you!",
                "confidence": 0.0,
                "matched_question": None,
                "category": None,
                "intent": "empty_input",
                "suggestions": self.get_sample_questions(3),
            }

        # 1. Check conversational intents first
        intent_response = self._check_conversational_intents(user_query_clean)
        if intent_response:
            return intent_response

        # 2. Perform Cosine Similarity matching over FAQ database
        best_item, max_score, top_candidates = self.match_question(user_query_clean)

        # 3. High/Acceptable Confidence Match
        if best_item and max_score >= self.similarity_threshold:
            # Related alternative suggestions (excluding matched one)
            suggestions = [
                cand["question"]
                for cand in top_candidates
                if cand["question"] != best_item["primary_question"]
            ][:2]

            return {
                "answer": best_item["answer"],
                "confidence": round(max_score, 3),
                "matched_question": best_item["primary_question"],
                "category": best_item["category"],
                "intent": "faq_match",
                "suggestions": suggestions,
            }

        # 4. Moderate Confidence (Did you mean...?)
        if top_candidates and max_score >= 0.15:
            suggested_questions = [cand["question"] for cand in top_candidates[:3]]
            return {
                "answer": (
                    "I am not entirely sure about that, but here are some related questions that might help:"
                ),
                "confidence": round(max_score, 3),
                "matched_question": None,
                "category": None,
                "intent": "partial_match",
                "suggestions": suggested_questions,
            }

        # 5. Fallback Response (Below threshold)
        return {
            "answer": (
                "I'm sorry, I couldn't find an answer matching your question. "
                "Could you please rephrase it, or ask about orders, shipping, returns, payments, or warranty?"
            ),
            "confidence": round(max_score, 3),
            "matched_question": None,
            "category": None,
            "intent": "fallback",
            "suggestions": self.get_sample_questions(3),
        }

    def get_sample_questions(self, count: int = 4) -> List[str]:
        """Returns a selection of sample questions from the FAQ dataset."""
        if not self.raw_faqs:
            return []
        sample_count = min(count, len(self.raw_faqs))
        samples = random.sample(self.raw_faqs, sample_count)
        return [faq["question"] for faq in samples]

    def get_all_questions(self) -> List[Dict[str, str]]:
        """Returns all questions with their category for catalog display."""
        return [
            {"category": faq["category"], "question": faq["question"]}
            for faq in self.raw_faqs
        ]

