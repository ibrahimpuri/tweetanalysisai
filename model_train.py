import pandas as pd
from sklearn.model_selection import train_test_split

# Load preprocessed data
df = pd.read_csv('preprocessed_data.csv')

# Split data into features (X) and target variable (y)
X = df.drop('target_column', axis=1)  # Assuming 'target_column' is the column containing the target variable
y = df['target_column']

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.naive_bayes import MultinomialNB

# Initialize the Naive Bayes classifier
model = MultinomialNB()

# Train the model on the training data
model.fit(X_train, y_train)

import joblib

# Save the trained model to a file
joblib.dump(model, 'naive_bayes_model.pkl')