"""
Gradio UI for Humor Detection
=============================
Interactive web interface for testing humor detection model

Author: AI Assistant
Date: 2024
"""

import gradio as gr
import torch
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class HumorDetector:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.tfidf_vectorizer = None
        self.traditional_model = None
        self.model_type = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the best model
        self.load_model()
    
    def clean_text(self, text):
        """Clean text using the same preprocessing as training"""
        # Convert to string if not already
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize and remove stopwords
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words and len(word) > 2]
        
        return ' '.join(filtered_text)
    
    def load_model(self):
        """Load the best trained model"""
        try:
            # Try to load DistilBERT model first
            if os.path.exists('./best_humor_model'):
                print("Loading DistilBERT model...")
                self.tokenizer = DistilBertTokenizer.from_pretrained('./best_humor_model')
                self.model = DistilBertForSequenceClassification.from_pretrained('./best_humor_model')
                self.model.to(self.device)
                self.model.eval()
                self.model_type = 'DistilBERT'
                print("DistilBERT model loaded successfully!")
                
            # Try to load traditional ML model
            elif os.path.exists('best_humor_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
                print("Loading traditional ML model...")
                self.traditional_model = joblib.load('best_humor_model.pkl')
                self.tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
                self.model_type = 'Traditional ML'
                print("Traditional ML model loaded successfully!")
                
            else:
                # Fallback: create a simple demo model
                print("No trained model found. Creating demo model...")
                self.model_type = 'Demo'
                
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_type = 'Demo'
    
    def predict_humor(self, text):
        """Predict if text is humorous"""
        if not text or text.strip() == "":
            return "Please enter some text to analyze.", 0.0, "WARNING"
        
        try:
            # Clean the input text
            cleaned_text = self.clean_text(text)
            
            if not cleaned_text:
                return "Text too short or contains no meaningful words.", 0.0, "WARNING"
            
            if self.model_type == 'DistilBERT':
                return self._predict_distilbert(cleaned_text, text)
            elif self.model_type == 'Traditional ML':
                return self._predict_traditional(cleaned_text, text)
            else:
                return self._predict_demo(text)
                
        except Exception as e:
            return f"Error during prediction: {str(e)}", 0.0, "ERROR"
    
    def _predict_distilbert(self, cleaned_text, original_text):
        """Predict using DistilBERT model"""
        # Tokenize
        inputs = self.tokenizer(
            cleaned_text, 
            return_tensors='pt', 
            truncation=True, 
            padding='max_length', 
            max_length=128
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(prediction, dim=-1).item()
            confidence = prediction[0][predicted_class].item()
        
        if predicted_class == 1:
            result = "HUMOR DETECTED!"
            emoji = "HUMOR"
        else:
            result = "NOT HUMOR"
            emoji = "NOT_HUMOR"
        
        return result, confidence, emoji
    
    def _predict_traditional(self, cleaned_text, original_text):
        """Predict using traditional ML model"""
        # Transform text using TF-IDF
        text_tfidf = self.tfidf_vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.traditional_model.predict(text_tfidf)[0]
        probability = self.traditional_model.predict_proba(text_tfidf)[0]
        confidence = max(probability)
        
        if prediction == 1:
            result = "HUMOR DETECTED!"
            emoji = "HUMOR"
        else:
            result = "NOT HUMOR"
            emoji = "NOT_HUMOR"
        
        return result, confidence, emoji
    
    def _predict_demo(self, text):
        """Demo prediction based on simple heuristics"""
        # Simple heuristics for demo
        humor_keywords = ['joke', 'funny', 'laugh', 'why', 'what do you call', 'pun', 'knock knock']
        text_lower = text.lower()
        
        humor_score = sum(1 for keyword in humor_keywords if keyword in text_lower)
        
        # Check for question marks (common in jokes)
        if '?' in text:
            humor_score += 1
        
        # Check for exclamation marks
        if '!' in text:
            humor_score += 0.5
        
        confidence = min(0.6 + humor_score * 0.1, 0.95)
        
        if humor_score >= 1:
            return "HUMOR DETECTED! (Demo Mode)", confidence, "HUMOR"
        else:
            return "NOT HUMOR (Demo Mode)", 1 - confidence, "NOT_HUMOR"

# Initialize the humor detector
detector = HumorDetector()

def analyze_humor(text):
    """Main function for Gradio interface"""
    result, confidence, emoji = detector.predict_humor(text)
    
    # Format confidence as percentage
    confidence_pct = f"{confidence * 100:.1f}%"
    
    # Create detailed output
    output = f"""
    ## {emoji} Prediction Result
    
    **{result}**
    
    **Confidence:** {confidence_pct}
    
    **Model Used:** {detector.model_type}
    
    ---
    
    **Original Text:** "{text}"
    """
    
    return output

# Create example texts
examples = [
    ["Why don't scientists trust atoms? Because they make up everything!"],
    ["I told my wife she was drawing her eyebrows too high. She looked surprised."],
    ["The stock market experienced significant volatility today."],
    ["Why did the scarecrow win an award? He was outstanding in his field!"],
    ["Climate change is a serious environmental concern."],
    ["What do you call a fake noodle? An impasta!"],
    ["The new policy will be implemented next quarter."],
    ["I'm reading a book about anti-gravity. It's impossible to put down!"]
]

# Create Gradio interface
with gr.Blocks(title="Humor Detection AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Humor Detection AI
    
    Enter any text and I'll tell you if it's humorous or not! This AI model has been trained to detect humor in text using advanced NLP techniques.
    
    **How it works:**
    - The model analyzes text patterns, word choices, and linguistic features
    - It provides a confidence score for its prediction
    - Try different types of text: jokes, news, stories, or random sentences!
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Enter text to analyze",
                placeholder="Type a joke, news headline, or any text here...",
                lines=3,
                max_lines=10
            )
            
            analyze_btn = gr.Button("Analyze Humor", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            gr.Markdown("### Quick Tips:")
            gr.Markdown("""
            - Try jokes, puns, or funny stories
            - Compare with serious news headlines
            - Experiment with different text lengths
            - The model works best with English text
            """)
    
    output = gr.Markdown(label="Analysis Result")
    
    # Add examples
    gr.Examples(
        examples=examples,
        inputs=text_input,
        label="Try these examples:"
    )
    
    # Connect the button to the function
    analyze_btn.click(
        fn=analyze_humor,
        inputs=text_input,
        outputs=output
    )
    
    # Also trigger on Enter key
    text_input.submit(
        fn=analyze_humor,
        inputs=text_input,
        outputs=output
    )
    
    gr.Markdown("""
    ---
    
    ### About the Model
    
    This humor detection system uses state-of-the-art NLP techniques including:
    - **Text Preprocessing:** Cleaning, tokenization, and stopword removal
    - **Feature Extraction:** TF-IDF and DistilBERT embeddings
    - **Machine Learning:** Logistic Regression, SVM, and fine-tuned DistilBERT
    
    The model was trained on a diverse dataset of humorous and non-humorous text to learn patterns that distinguish humor from regular text.
    """)

if __name__ == "__main__":
    print("Starting Humor Detection Gradio App...")
    print(f"Model type: {detector.model_type}")
    print("Open the URL below in your browser to use the app!")
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )