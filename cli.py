"""
Command Line Interface (CLI) for CodeAlpha FAQ Chatbot
Run this script to chat with the bot directly from your terminal.
"""

import sys
from chatbot.faq_engine import FAQChatbot


def print_banner():
    banner = """
============================================================
       🤖 CodeAlpha AI Internship - FAQ Chatbot 🤖
------------------------------------------------------------
 * NLP Pipeline: NLTK (Tokenization, Lemmatization, Stopwords)
 * Match Method: TF-IDF Vectorizer + Cosine Similarity
 * Type 'exit', 'quit', or 'bye' to exit.
 * Type 'help' to see sample questions.
============================================================
"""
    print(banner)


def main():
    print_banner()
    try:
        bot = FAQChatbot()
        print("Bot initialized successfully! Ready for your questions.\n")
    except Exception as e:
        print(f"[Error] Failed to initialize FAQ Chatbot: {e}")
        sys.exit(1)

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            # Process user input
            response = bot.get_response(user_input)

            # Display response
            print(f"\nBot: {response['answer']}")

            # Show metadata if it was an FAQ match
            if response.get("intent") == "faq_match":
                print(f"     [Category: {response['category']} | Matched: \"{response['matched_question']}\" | Confidence: {response['confidence'] * 100:.1f}%]")

            # Show suggestions if available
            suggestions = response.get("suggestions", [])
            if suggestions:
                print("     💡 Suggested Questions:")
                for i, sug in enumerate(suggestions, 1):
                    print(f"        {i}. {sug}")

            print("-" * 60 + "\n")

            if response.get("intent") == "farewell":
                break

        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye! Have a great day!")
            break


if __name__ == "__main__":
    main()
