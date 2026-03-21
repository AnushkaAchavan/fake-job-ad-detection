import numpy as np
import pandas as pd
import seaborn as sns

import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/MyDrive/fake_job_postings (1).csv')
df.head()

import matplotlib.pyplot as plt
print("\nFraudulent distribution:")
print(df['fraudulent'].value_counts(normalize=True))

# Highly imbalanced → we will use class_weight later
plt.figure(figsize=(6,4))
sns.countplot(x='fraudulent', data=df)
plt.title('Real vs Fraudulent Job Postings')
plt.show()

text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
cat_columns = ['location', 'department', 'salary_range', 'employment_type',
               'required_experience', 'required_education', 'industry', 'function']

binary_columns = ['telecommuting', 'has_company_logo', 'has_questions']

for col in text_columns:
    df[col] = df[col].fillna('')
df['full_text'] = df[text_columns].agg(' '.join, axis=1)

for col in cat_columns:
    df[col] = df[col].fillna('missing')
for col in binary_columns:
    df[col] = df[col].fillna(0)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http[s]?://\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word)
              for word in tokens
              if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

df['cleaned_text'] = df['full_text'].apply(clean_text)
