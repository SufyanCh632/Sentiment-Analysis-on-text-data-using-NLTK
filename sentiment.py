# ==============================
# Movie Review Sentiment Analysis
# Improved Version
# ==============================

# Import Libraries
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
from nltk.probability import FreqDist

# Download required datasets (only first time)
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')

# ------------------------------
# Prepare Stopwords (Optimized)
# ------------------------------
stop_words = set(stopwords.words('english'))

# ------------------------------
# Load and Prepare Documents
# ------------------------------
documents = [
    (list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

# Shuffle dataset
random.shuffle(documents)

# ------------------------------
# Build Frequency Distribution
# ------------------------------
all_words = []

for word in movie_reviews.words():
    word = word.lower()
    if word.isalpha() and word not in stop_words:
        all_words.append(word)

# Get most common 2000 words
all_words_freq = FreqDist(all_words)
word_features = list(all_words_freq.keys())[:2000]

# ------------------------------
# Feature Extraction Function
# ------------------------------
def extract_features(document_words):
    document_words = [w.lower() for w in document_words if w.isalpha()]
    document_words = set(document_words)

    features = {}
    for word in word_features:
        features[word] = (word in document_words)

    return features

# ------------------------------
# Create Feature Sets
# ------------------------------
featuresets = [(extract_features(d), c) for (d, c) in documents]

# Train-Test Split (80-20)
train_set = featuresets[:1600]
test_set = featuresets[1600:]

# ------------------------------
# Train Classifier
# ------------------------------
classifier = NaiveBayesClassifier.train(train_set)

# ------------------------------
# Evaluate Model
# ------------------------------
accuracy = nltk_accuracy(classifier, test_set)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")

print("Most Informative Features:")
classifier.show_most_informative_features(10)

# ------------------------------
# Sentiment Prediction Function
# ------------------------------
def analyze_sentiment(text):
    tokens = nltk.word_tokenize(text)
    features = extract_features(tokens)
    return classifier.classify(features)

# ------------------------------
# Test Custom Sentences
# ------------------------------
test_sentences = [
    "This movie is absolutely fantastic! The acting and story were amazing.",
    "I hated this movie. It was a complete waste of time.",
    "The plot was dull but the performances were great.",
    "I have mixed feelings. It was okay, not great but not terrible."
]

print("\nCustom Predictions:\n")
for sentence in test_sentences:
    print(f"Sentence: {sentence}")
    print(f"Predicted Sentiment: {analyze_sentiment(sentence)}\n")