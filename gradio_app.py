import gradio as gr
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')


class HumorDetector:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.traditional_model = None
        self.model_type = None
        
        self.load_model()

    def load_model(self):
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))

            model_path = os.path.join(base_path, 'best_humor_model.pkl')
            vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer.pkl')

            print("Loading Traditional ML model...")

            self.traditional_model = joblib.load(model_path)
            self.tfidf_vectorizer = joblib.load(vectorizer_path)

            self.model_type = 'Traditional ML'
            print("SUCCESS Model loaded successfully!")

        except Exception as e:
            print(f"ERROR loading model: {e}")
            self.model_type = 'Demo'

    def clean_text(self, text):
        text = str(text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())

        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        filtered = [w for w in words if w not in stop_words and len(w) > 2]

        return ' '.join(filtered)

    def predict_humor(self, text):
        if not text.strip():
            return "Enter text", 0.0, "WARNING"

        try:
            cleaned = self.clean_text(text)

            if self.model_type == 'Traditional ML':
                vec = self.tfidf_vectorizer.transform([cleaned])
                pred = self.traditional_model.predict(vec)[0]
                prob = self.traditional_model.predict_proba(vec)[0]
                conf = max(prob)

                if pred == 1:
                    return "HUMOR DETECTED!", conf, "HUMOR"
                else:
                    return "NOT HUMOR", conf, "NOT_HUMOR"

            else:
                # fallback demo
                if "why" in text.lower() or "joke" in text.lower():
                    return "HUMOR (Demo)", 0.7, "HUMOR"
                else:
                    return "NOT HUMOR (Demo)", 0.6, "NOT_HUMOR"

        except Exception as e:
            return f"Error: {str(e)}", 0.0, "ERROR"


detector = HumorDetector()


def analyze_humor(text):
    result, confidence, emoji = detector.predict_humor(text)

    return f"""
## {emoji}

**{result}**

Confidence: {confidence*100:.2f}%

Model: {detector.model_type}
"""


with gr.Blocks() as demo:
    gr.Markdown("# Humor Detection AI")

    text_input = gr.Textbox(label="Enter text")
    btn = gr.Button("Analyze")

    output = gr.Markdown()

    btn.click(fn=analyze_humor, inputs=text_input, outputs=output)

if __name__ == "__main__":
    demo.launch()