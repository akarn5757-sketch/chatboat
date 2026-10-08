"""
Unit tests for FAQ Chatbot & NLP Preprocessing
"""

import unittest
from chatbot.preprocessor import TextPreprocessor
from chatbot.faq_engine import FAQChatbot


class TestTextPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = TextPreprocessor()

    def test_clean_text(self):
        sample = "Can't you track my order??? #1234!"
        cleaned = self.preprocessor.clean_text(sample)
        self.assertIn("cannot", cleaned)
        self.assertNotIn("?", cleaned)
        self.assertNotIn("#", cleaned)

    def test_tokenize(self):
        sample = "How do I return a product?"
        tokens = self.preprocessor.tokenize(sample)
        self.assertIn("how", tokens)
        self.assertIn("return", tokens)
        self.assertIn("product", tokens)

    def test_lemmatization(self):
        # Words like 'returns', 'deliveries' should be stemmed/lemmatized
        tokens = self.preprocessor.preprocess_tokens("Items are being returned")
        # 'returned' should be lemmatized to 'return'
        self.assertIn("return", tokens)


class TestFAQChatbot(unittest.TestCase):
    def setUp(self):
        self.bot = FAQChatbot()

    def test_greeting_intent(self):
        response = self.bot.get_response("hello")
        self.assertEqual(response["intent"], "greeting")
        self.assertEqual(response["confidence"], 1.0)
        self.assertTrue(len(response["answer"]) > 0)

    def test_farewell_intent(self):
        response = self.bot.get_response("bye")
        self.assertEqual(response["intent"], "farewell")
        self.assertEqual(response["confidence"], 1.0)

    def test_order_tracking_query(self):
        response = self.bot.get_response("How can I track my shipment package?")
        self.assertEqual(response["intent"], "faq_match")
        self.assertGreater(response["confidence"], 0.3)
        self.assertIn("track", response["answer"].lower())

    def test_refund_query(self):
        response = self.bot.get_response("When will my money be refunded?")
        self.assertEqual(response["intent"], "faq_match")
        self.assertGreater(response["confidence"], 0.3)
        self.assertIn("refund", response["answer"].lower())

    def test_irrelevant_query_fallback(self):
        response = self.bot.get_response("What is the recipe for chocolate cake?")
        self.assertEqual(response["intent"], "fallback")
        self.assertLess(response["confidence"], 0.25)


if __name__ == "__main__":
    unittest.main()
