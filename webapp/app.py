from flask import Flask, render_template, request
import numpy as np
import os
import re
from sklearn import base
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')

app = Flask(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to model and tokenizer
MODEL_PATH = os.path.join(base_dir, 'model.h5')
TOKENIZER_PATH = os.path.join(base_dir, 'tokenizer.pkl')

# Load tokenizer
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)


# Enable eager execution in TensorFlow
tf.compat.v1.enable_eager_execution()


# Check if eager execution is enabled
assert tf.executing_eagerly(), "Eager execution is not enabled!"

# Initialize model and graph
# Initialize model
model = tf.keras.models.load_model(MODEL_PATH)

# Text preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove emojis
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Stemming
    ps = PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])

    return text

@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = None
    probability = None
    text = None

    if request.method == 'POST':
        text = request.form['text']
        
        # Preprocess the input text
        processed_text = preprocess_text(text)

        # Convert text to sequences
        tw = tokenizer.texts_to_sequences([processed_text])
        tw = sequence.pad_sequences(tw, maxlen=25)

       

        # Make prediction
        probabilities = model.predict(tw)[0]
        prediction = np.argmax(probabilities) 

            # Determine sentiment label
        if prediction == 0:
            sentiment = 'Negative 😞'
        elif prediction == 1:
            sentiment = 'Neutral 😐'
        else:
            sentiment = 'Positive 😊'
        
        probability = np.max(probabilities)

    return render_template('index.html', text=text, sentiment=sentiment, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)