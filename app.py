from flask import Flask, request, jsonify, render_template
import os
import uuid
import logging
from werkzeug.utils import secure_filename
from langdetect import detect, DetectorFactory
from dotenv import load_dotenv

import requests
from transformers import pipeline
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO)
DetectorFactory.seed = 0

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Translation models
translate_en = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translate_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

# Captioning models
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Functions
def auto_translate_input(text):
    lang = detect(text)
    if lang == "fr":
        translated = translate_en(text)[0]['translation_text']
        return translated, "fr"
    return text, "en"

def auto_translate_output(response, target_lang):
    if target_lang == "fr":
        return translate_fr(response)[0]['translation_text']
    return response

def ask_openrouter(prompt, lang="en"):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "chatgpt-clone",
        "Content-Type": "application/json"
    }
    messages = [
        {"role": "system", "content": "You are a helpful multilingual assistant."},
        {"role": "user", "content": prompt}
    ]
    body = {
        "model": "mistralai/mixtral-8x7b",
        "messages": messages,
        "temperature": 0.7
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f"LLM error: {e}")
        return "Sorry, I couldn't process that right now."

def image_caption(file_path):
    try:
        image = Image.open(file_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        logging.error(f"Image captioning error: {e}")
        return "This looks like an image, but I couldn't understand it."

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        if 'image' in request.files:
            file = request.files['image']
            if not file or not file.mimetype.startswith('image/'):
                return jsonify({"response": "No valid image selected"})
            filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            caption = image_caption(filepath)
            response = ask_openrouter(f"What can you say about this image? Description: {caption}")

            os.remove(filepath)
            return jsonify({"response": response})

        elif 'message' in request.form:
            user_input = request.form.get("message")
            if not user_input:
                return jsonify({"response": "Please enter a message."})

            translated_input, detected_lang = auto_translate_input(user_input)
            llm_response = ask_openrouter(translated_input, lang=detected_lang)
            final_response = auto_translate_output(llm_response, detected_lang)

            return jsonify({"response": final_response})

        return jsonify({"response": "Please send a message or an image."})

    except Exception as e:
        logging.error(f"Chat error: {e}")
        return jsonify({"response": "Something went wrong."})

if __name__ == "__main__":
    app.run(debug=True)