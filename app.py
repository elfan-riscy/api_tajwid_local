from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
from utils.mfcc_extractor import extract_mfcc

app = Flask(__name__)
UPLOAD_FOLDER = 'audio'
MODEL_PATH = 'model/model_tajwid_benar_salah.h5'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

label_list = ['Benar', 'Salah']

# ✅ Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model berhasil dimuat")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    model = None

@app.route('/')
def index():
    return "✅ API Tajwid Siap Digunakan!"

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model belum dimuat'}), 500

    if 'audio' not in request.files or 'teks' not in request.form:
        return jsonify({'error': '❌ File audio dan teks target harus disertakan'}), 400

    file = request.files['audio']
    teks_target = request.form['teks'].strip()
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    print(f"🎤 File diterima: {file_path}")

    try:
        mfcc = extract_mfcc(file_path)
        mfcc = mfcc.reshape(1, 40, 100, 1)

        pred = model.predict(mfcc)[0]
        pred_index = int(np.argmax(pred))
        confidence = float(pred[pred_index])
        label = label_list[pred_index]

        # 🎯 Feedback berdasarkan label dan confidence
        if label == "Salah":
            feedback = "❌ Bacaan SALAH. Ulangi!"
        elif confidence < 0.6:
            feedback = "⚠️ Bacaan Kurang Tepat"
        elif confidence < 0.7:
            feedback = "⚠️ Bacaan Hampir Tepat"
        elif confidence < 0.8:
            feedback = "✅ Bacaan Hampir Benar"
        else:
            feedback = "✅ Bacaan BENAR"

        # 🧠 Mapping feedbackState untuk AR Unity
        if label == "Salah":
            feedbackState = "salah"
        elif confidence < 0.6:
            feedbackState = "kurangtepat"
        elif confidence < 0.7:
            feedbackState = "hampirtepat"
        elif confidence < 0.8:
            feedbackState = "hampirbenar"
        else:
            feedbackState = "benar"

        return jsonify({
            'label': label,
            'confidence': round(confidence * 100, 2),
            'feedback': feedback,
            'feedbackState': feedbackState,
            'target_teks': teks_target
        })

    except Exception as e:
        return jsonify({'error': f'❌ Error saat prediksi: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
