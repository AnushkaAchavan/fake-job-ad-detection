!pip install textblob spacy

import pandas as pd
import numpy as np
import re
import spacy
from textblob import TextBlob
from collections import Counter

nlp = spacy.load("en_core_web_sm")

#Lexical features
def lexical_features(text):
    words = text.split()
    sentences = text.split('.')

    return {
        "word_count": len(words),
        "avg_sentence_length": len(words) / (len(sentences) + 1),

        # Suspicious keywords
        "suspicious_words": sum(1 for w in words if w in [
            "earn", "quick", "money", "urgent", "no", "experience", "required"
        ]),

        # Capitalization ratio
        "capital_ratio": sum(1 for w in words if w.isupper()) / (len(words) + 1),

        # Punctuation count
        "exclamation_count": text.count('!'),
        "question_count": text.count('?')
    }

#Syntatic features
def syntactic_features(text):
    doc = nlp(text)

    pos_counts = Counter([token.pos_ for token in doc])

    nouns = pos_counts['NOUN']
    verbs = pos_counts['VERB']

    return {
        "noun_count": nouns,
        "verb_count": verbs,
        "noun_verb_ratio": nouns / (verbs + 1),
        "sentence_count": len(list(doc.sents))
    }

#Semantic Features
def semantic_features(text):
    blob = TextBlob(text)

    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity
    }

#Structural Features
def structural_features(text):
    return {
        # Email detection
        "has_email": 1 if re.search(r'\S+@\S+', text) else 0,

        # Phone number detection
        "has_phone": 1 if re.search(r'\d{10}', text) else 0,

        # Salary mention
        "has_salary": 1 if re.search(r'\$\d+|\d+\s?USD|\d+\s?rs', text.lower()) else 0,

        # Company mention
        "has_company": 1 if "company" in text.lower() else 0
    }

#Combine all features
def extract_linguistic_features(text):
    features = {}

    features.update(lexical_features(text))
    features.update(syntactic_features(text))
    features.update(semantic_features(text))
    features.update(structural_features(text))

    return features

#APPLY TO DATASET
linguistic_df = df['description'].apply(extract_linguistic_features)
linguistic_df = pd.DataFrame(linguistic_df.tolist())

linguistic_df.head()

#MERGE WITH TF-IDF FEATURES
tfidf_df = pd.DataFrame(text_features.toarray(), columns=tfidf.get_feature_names_out())
final_features = pd.concat([tfidf_df, linguistic_df], axis=1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = final_features
y = df['fraudulent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
