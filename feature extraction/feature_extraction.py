tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
text_features = tfidf.fit_transform(df['text'])
df_cat = pd.get_dummies(df[cat_columns], drop_first=True)
df_num = df[num_cols]
df['desc_length'] = df['description'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
df['desc_length'] = df['description'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
df[f'has_{word}'] = ...
X = hstack([text_features,
             df_cat.values,
             df_num.values,
             df[['desc_length','word_count']].values])
