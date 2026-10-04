import streamlit as st
import joblib

# Load trained model and TF-IDF vectorizer
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📱"
)

# Title
st.title("📱 Spam Message Detector")

st.write(
    "Enter an SMS message below and the machine learning model "
    "will predict whether it is Spam or Ham."
)

# Text input
message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You won a free prize!"
)

# Prediction button
if st.button("Predict"):
    
    if message.strip() == "":
        st.warning("Please enter a message first.")
    
    else:
        # Convert text into TF-IDF features
        message_tfidf = vectorizer.transform([message])

        # Make prediction
        prediction = model.predict(message_tfidf)[0]

        # Display result
        if prediction == "spam":
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is HAM (Normal)")