import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

df = pd.read_csv('tweets.csv')
df = df[['text_column', 'label_column']]
df.dropna(inplace=True)
df['text_column'] = df['text_column'].str.lower()
df['text_column'] = df['text_column'].str.replace('[^\w\s]', '')
df['text_column'] = df['text_column'].apply(lambda x: x.split())

count_vectorizer = CountVectorizer()
X_counts = count_vectorizer.fit_transform(df['text_column'])

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['label_column'], test_size=0.2, random_state=42)
