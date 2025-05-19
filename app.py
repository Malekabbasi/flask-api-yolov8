from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
model = YOLO("best.pt")

@app.route('/')
def home():
    return jsonify({"message": "✅ YOLOv8 API on Render is running"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "❌ صورة غير موجودة"}), 400
    try:
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        results = model(image)
        boxes = results[0].boxes
        names = results[0].names

        if len(boxes) > 0:
            class_id = int(boxes.cls[0].item())
            label = names[class_id]
            confidence = float(boxes.conf[0])
            return jsonify({
                "diagnosis": label,
                "confidence": confidence
            })
        else:
            return jsonify({
                "diagnosis": "✅ لا توجد إصابة واضحة",
                "confidence": 1.0
            })
    except Exception as e:
        return jsonify({"error": f"❌ خطأ داخلي: {str(e)}"}), 500

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render يعطي PORT ديناميكي
    app.run(host='0.0.0.0', port=port)

