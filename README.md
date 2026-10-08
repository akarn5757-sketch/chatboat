# CodeAlpha AI Internship — Task 2: Chatbot for FAQs

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLP-NLTK-green.svg)](https://www.nltk.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/UI-Flask%20Web%20App-red.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An intelligent conversational FAQ Chatbot built for **CodeAlpha's Artificial Intelligence Internship (Task 2)**. The chatbot preprocesses user inquiries with **NLTK**, computes semantic text representations with **TF-IDF Vectorization**, and matches inquiries to the most relevant FAQ using **Cosine Similarity**. It features both a modern **interactive Web Chat UI** and a **Command-Line Interface (CLI)**.

---

## 📌 Project Overview & Task Alignment

According to the official CodeAlpha assignment criteria:

| Requirement | Implementation Detail | Status |
| :--- | :--- | :---: |
| **Collect FAQs** | Comprehensive FAQ knowledge base (`data/faqs.json`) covering orders, tracking, refunds, returns, payments, security, and warranties with question pattern variations. | ✅ Completed |
| **Text Preprocessing** | NLTK-based cleaning pipeline: lowercasing, contraction handling, tokenization (`nltk.tokenize`), stopword filtering (`nltk.corpus.stopwords`), and lemmatization (`nltk.stem.WordNetLemmatizer`). | ✅ Completed |
| **Similarity Matching** | TF-IDF n-gram vectorization (`TfidfVectorizer`) combined with Cosine Similarity (`sklearn.metrics.pairwise.cosine_similarity`) and dynamic confidence thresholding. | ✅ Completed |
| **Response Generation** | Displays the best matching answer, category, match confidence percentage, and suggested alternative questions. | ✅ Completed |
| **Interactive UI** | Sleek, responsive web chat interface built with **Flask**, **HTML5**, **CSS3**, and **JavaScript**, plus a terminal CLI. | ✅ Completed |

---

## 🏗️ Architecture & NLP Pipeline

```
[User Input Query]
       │
       ▼
[1. Intent & Small-Talk Check] ──► (Greeting / Farewell / Help / Gratitude)
       │ (if not chit-chat)
       ▼
[2. NLTK Text Preprocessor]
  ├── Lowercase & Contraction Normalization
  ├── Regex Cleaning (strip punctuation & noise)
  ├── Word Tokenization (`nltk.word_tokenize`)
  ├── Stopword Filtering (`nltk.corpus.stopwords`)
  └── Morphological Lemmatization (`WordNetLemmatizer`)
       │
       ▼
[3. Scikit-Learn TF-IDF Vectorizer]
  └── Transform query into unigram & bigram TF-IDF feature space
       │
       ▼
[4. Cosine Similarity Matcher]
  └── Compute pairwise cosine similarity against all FAQ question patterns
       │
       ▼
[5. Confidence Evaluator]
  ├── Score >= 0.25 : Direct Match (Return Answer + Confidence + Related FAQs)
  ├── 0.15 <= Score < 0.25 : Partial Match ("Did you mean...?" suggestions)
  └── Score < 0.15 : Fallback ("I couldn't find an answer... please rephrase")
```

---

## 📁 Repository Structure

```
Chatbot for FAQs/
├── data/
│   └── faqs.json               # FAQ database with questions, patterns, & answers
├── chatbot/
│   ├── __init__.py             # Module exports
│   ├── preprocessor.py         # NLTK tokenization, cleaning, lemmatization
│   └── faq_engine.py           # TF-IDF, Cosine Similarity & match logic
├── templates/
│   └── index.html              # Modern, responsive Web Chat interface
├── static/
│   ├── css/
│   │   └── style.css           # Styling, themes, animations, message bubbles
│   └── js/
│       └── app.js              # Fetch API client, typing animation, chips
├── app.py                      # Flask web server & REST API
├── cli.py                      # Interactive terminal interface
├── test_chatbot.py             # Automated unit tests
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore configuration
└── README.md                   # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

### 2. Installation
Open your terminal in the project directory and install the required dependencies:

```bash
pip install -r requirements.txt
```

*(Note: Required NLTK datasets like `punkt`, `stopwords`, and `wordnet` are downloaded automatically upon first execution).*

---

## 💻 Running the Chatbot

You can run the project in two different modes:

### Option A: Modern Web UI (Recommended)
Run the Flask server:
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```
**Web Features:**
- Real-time conversational interface with typing indicator.
- Knowledge Base sidebar showcasing all indexed FAQs.
- Confidence score indicator (e.g. `82% match`) and category badge for every answer.
- Interactive question suggestion chips that can be clicked to ask follow-up questions.
- Responsive design tailored for desktop and mobile browsers.

---

### Option B: Command-Line Interface (CLI)
To run directly in your terminal:
```bash
python cli.py
```

**Example Terminal Session:**
```text
============================================================
       🤖 CodeAlpha AI Internship - FAQ Chatbot 🤖
------------------------------------------------------------
 * NLP Pipeline: NLTK (Tokenization, Lemmatization, Stopwords)
 * Match Method: TF-IDF Vectorizer + Cosine Similarity
 * Type 'exit', 'quit', or 'bye' to exit.
============================================================

Bot initialized successfully! Ready for your questions.

You: How can I track my shipment package?

Bot: Once your order ships, we send a tracking number via email and SMS. You can also track your shipment directly in the 'My Orders' section of your account or enter your tracking ID on our tracking portal.
     [Category: Shipping & Tracking | Matched: "How do I track my order?" | Confidence: 73.5%]
     💡 Suggested Questions:
        1. How long does shipping take and what are the delivery charges?
        2. Can I cancel or modify my order after placing it?
------------------------------------------------------------
```

---

## 🧪 Running Unit Tests

Run the test suite to verify NLP preprocessing and similarity matching:

```bash
python -m unittest test_chatbot.py
```

Expected output:
```text
Ran 8 tests in 2.29s
OK
```

---

## 🌐 REST API Endpoints

The Flask application also exposes RESTful endpoints:

### `POST /api/chat`
Send a user inquiry and get an intelligent response.

**Request Body:**
```json
{
  "message": "Is it safe to pay with credit card?"
}
```

**Response:**
```json
{
  "success": true,
  "query": "Is it safe to pay with credit card?",
  "response": {
    "answer": "Yes, absolutely. All transactions are encrypted using 256-bit SSL encryption and processed through PCI-DSS compliant payment gateways. We never store your full card details.",
    "confidence": 0.619,
    "matched_question": "Is it safe to use my credit card on your website?",
    "category": "Payments & Security",
    "intent": "faq_match",
    "suggestions": [
      "What payment methods do you accept?",
      "How and when will I receive my refund?"
    ]
  }
}
```

### `GET /api/faqs`
Retrieves the full list of indexed FAQs.

### `GET /api/health`
Health check status.

---

## 📤 Submission Instructions for CodeAlpha Interns

1. **GitHub Repository**:
   - Initialize git and push this codebase to a repository named:
     ```
     CodeAlpha_Chatbot_for_FAQs
     ```
   - Commands to push to GitHub:
     ```bash
     git init
     git add .
     git commit -m "Complete Task 2: Chatbot for FAQs"
     git branch -M main
     git remote add origin https://github.com/<your-username>/CodeAlpha_Chatbot_for_FAQs.git
     git push -u origin main
     ```
2. **LinkedIn Post**:
   - Share a short video explanation or screen demo of the chatbot running in the Web UI and terminal.
   - Include the GitHub repository link.
   - Tag `@CodeAlpha` and mention `#CodeAlpha #AI #MachineLearning #NLP`.
3. **Submission Form**:
   - Submit the repository link and LinkedIn post URL in the CodeAlpha submission form.

---

## 📄 License
This project is open-source under the MIT License.
