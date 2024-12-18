from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
import os
from random import randint
import time


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
SAVE_FILENAME = 'meme.png'

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
def load_model():
    pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], SAVE_FILENAME)
        file.save(filepath)
        return jsonify({'filename': SAVE_FILENAME})
    return jsonify({'error': 'No file uploaded'}), 400

@app.route('/evaluate')
def evaluate():
    # model.eval()
    result = bool(randint(0, 1))
    return render_template("index.html", hateful=result)

if __name__ == '__main__':
    model = load_model()
    app.run(debug=True)
