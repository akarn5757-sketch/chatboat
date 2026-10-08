"""
Flask Web Application & REST API for CodeAlpha FAQ Chatbot
Provides an interactive Chat UI and RESTful API endpoints.
"""

import os
from flask import Flask, jsonify, render_template, request
from chatbot.faq_engine import FAQChatbot

app = Flask(__name__)

# Initialize FAQ Chatbot engine
bot = FAQChatbot()


@app.route("/")
def index():
    """Renders the modern chat interface."""
    sample_questions = bot.get_sample_questions(4)
    all_faqs = bot.get_all_questions()
    return render_template("index.html", sample_questions=sample_questions, faqs=all_faqs)


@app.route("/api/chat", methods=["POST"])
def chat():
    """API endpoint to send user query and receive chatbot response."""
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({
            "success": False,
            "error": "Message cannot be empty."
        }), 400

    response_data = bot.get_response(user_message)

    return jsonify({
        "success": True,
        "query": user_message,
        "response": response_data
    })


@app.route("/api/faqs", methods=["GET"])
def get_faqs():
    """Returns the entire list of indexed FAQ questions."""
    return jsonify({
        "success": True,
        "total": len(bot.raw_faqs),
        "faqs": bot.raw_faqs
    })


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "CodeAlpha FAQ Chatbot",
        "faqs_loaded": len(bot.raw_faqs)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting FAQ Chatbot server at http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
