import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import numpy as np

# Custom CSS for Dark Mode Styling
st.markdown(
    """
    <style>
    /* Hide the Streamlit "link" icon */
    .stTextArea textarea:focus ~ .css-1yqiy10 {
        visibility: hidden;
    }

    /* Main background */
    .main {
        background-color: #121212;  /* Dark background */
        color: white;  /* White text color */
    }

    /* Title style */
    h1 {
        font-family: 'Arial', sans-serif;
        font-size: 40px;
        color: #1E90FF;  /* Blue color */
        text-align: center;
        margin-top: 20px;
    }

    /* Subtitle style */
    h3 {
        font-family: 'Arial', sans-serif;
        font-size: 20px;
        color: #1E90FF;  /* Blue color */
        text-align: center;
        margin-top: 10px;
    }

    /* Button style */
    .stButton>button {
        background-color: #1E90FF;  /* Blue button */
        color: white;
        font-size: 18px;
        border-radius: 12px;
        padding: 12px 25px;
        width: 100%;
        border: none;
    }

    /* Text area style */
    .stTextArea>textarea {
        background-color: #333333;  /* Dark background for text area */
        color: white;  /* White text */
        border: 2px solid #1E90FF;  /* Blue border */
        border-radius: 10px;
        font-size: 16px;
        padding: 10px;
    }

    /* Bar graph customization */
    .matplotlib {
        background-color: #121212;
        border-radius: 10px;
        padding: 20px;
    }

    /* Footer style */
    footer {
        visibility: hidden;
    }

    /* Style for highlighted spam words */
    .highlighted {
        color: red;
        font-weight: bold;
    }

    /* Style for result text to make it bold */
    .stResult {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }

    </style>
    """, unsafe_allow_html=True
)

# Function to preprocess text
def transform_sentence(text):
    text = text.lower()  # Lowercase
    text = nltk.word_tokenize(text)  # Tokenization

    temp = []  # Removing special characters
    for word in text:
        if word.isalnum():
            temp.append(word)
    text = temp

    stopw = stopwords.words('english')
    punw = string.punctuation

    temp = [word for word in text if word not in stopw and word not in punw]  # Remove stopwords and punctuation

    ps = PorterStemmer()
    text = [ps.stem(word) for word in temp]  # Stemming

    return " ".join(text)

# Load model and vectorizer
cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app title
st.title("Enhanced Email Spam Classifier")

# Input field for email text
input_email = st.text_area("Enter the message", height=200)

# Prediction button
if st.button('Predict'):
    if input_email != "":
        # Preprocess the email
        transformed_email = transform_sentence(input_email)

        # Vectorize the transformed email
        vector_input = cv.transform([transformed_email])

        # Predict spam or not spam
        result = model.predict(vector_input)[0]
        probabilities = model.predict_proba(vector_input)[0]

        # Highlight spam words in the original email
        spam_words = set(word for word in transformed_email.split() if word in cv.get_feature_names_out())
        highlighted_email = [
            f"<span class='highlighted'>{word}</span>" if word in spam_words else word for word in input_email.split()
        ]
        st.subheader("Highlighted Spam Words:")
        st.markdown(" ".join(highlighted_email), unsafe_allow_html=True)

        # Bar graph visualization of probabilities
        st.subheader("Spam Prediction Probability:")
        fig, ax = plt.subplots()
        categories = ['Not Spam', 'Spam']
        ax.bar(categories, probabilities, color=['green', 'red'])
        plt.ylabel('Probability')
        plt.title('Prediction Probability')
        st.pyplot(fig)

        # Display result in bold
        if result == 1:
            st.markdown("<p class='stResult'>This email is <strong>Spam 🚨</strong></p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='stResult'>This email is <strong>Not Spam ✅</strong></p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='stResult'>Please enter a message to classify.</p>", unsafe_allow_html=True)

