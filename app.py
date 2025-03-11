from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask Face Detection API is Running!"

@app.route('/detect', methods=['POST'])
def detect():
    # รับไฟล์ภาพจาก request
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    # แปลงไฟล์ภาพเป็น OpenCV format
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # โหลดโมเดล Haarcascade สำหรับตรวจจับใบหน้า
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # แปลงภาพเป็นขาวดำ
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้า
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # แปลงผลลัพธ์เป็น JSON
    result = {"faces_detected": len(faces)}
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)