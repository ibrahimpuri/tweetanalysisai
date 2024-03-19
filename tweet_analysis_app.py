import tkinter as tk
from tkinter import messagebox
import joblib

# Load the trained model
model = joblib.load(
    'model_train.py')  # Replace 'your_model_filename.pkl' with the actual filename of your trained model


def preproccessed_data(text):
    # Add your preprocessing steps here
    # For example: convert text to lowercase, remove punctuation, tokenize text, etc.
    return text


def predict_sentiment():
    text = text_entry.get("1.0", "end").strip()
    if text:
        # Preprocess the text
        text = preproccessed_data(text)

        # Use the trained model to predict sentiment
        prediction = model.predict([text])[0]
        if prediction == 1:
            sentiment = "Positive"
        else:
            sentiment = "Negative"

        # Display the sentiment prediction to the user
        messagebox.showinfo("Sentiment Prediction", f"The sentiment of the text is {sentiment}")
    else:
        messagebox.showwarning("Error", "Please enter some text.")


# Create GUI
root = tk.Tk()
root.title("Sentiment Analysis")

label = tk.Label(root, text="Enter text:")
label.pack()

text_entry = tk.Text(root, height=5, width=50)
text_entry.pack()

predict_button = tk.Button(root, text="Predict Sentiment", command=predict_sentiment)
predict_button.pack()

root.mainloop()