import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
corpus = [
    "hello",
    "hi",
    "how are you",
    "what is ai",
    "what is nlp",
    "what is python",
    "ai means artificial intelligence",
    "nlp means natural language processing",
    "python is a programming language",
    "bye",
    "thank you"
]

responses = {
    "hello": "Hello! How can I help you?",
    "hi": "Hi there!",
    "how are you": "I am fine ",
    "what is ai": "AI stands for Artificial Intelligence.",
    "what is nlp": "NLP allows computers to understand human language.",
    "what is python": "Python is used for AI, ML, and web development.",
    "bye": "Goodbye! ",
    "thank you": "You're welcome "
}

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

def chatbot_reply(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    index = similarity.argmax()

    if similarity[0][index] < 0.2:
        return "Sorry, I didn't understand that."
    return responses[corpus[index]]
print(" Simple AI Chatbot is running (type 'bye' to exit)")

while True:
    user = input("You: ").lower()

    if user == "bye":
        print(" Bot: Goodbye! ")
        break

    print(" Bot:", chatbot_reply(user))
