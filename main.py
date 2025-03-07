import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import cv2
import cloudinary
import cloudinary.uploader
import cloudinary.api
from flask import Flask, request, jsonify
import numpy as np
from io import BytesIO

port = int(os.environ.get("PORT", 443))

app = Flask(__name__)


# Cấu hình Cloudinary
cloudinary.config(
    cloud_name='dphfcojlc',
    api_key='133562825661752',
    api_secret='f1UBV7B7jR03USGeqgeuRuXKdnI'
)

# Tải cascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Hàm xử lý ảnh và lưu vào Cloudinary
def process_and_upload_face(frame):
    # Chuyển ảnh sang màu xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong ảnh
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Cắt khuôn mặt
        face = frame[y:y+h, x:x+w]
        
        # Vẽ hình chữ nhật quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Chuyển ảnh khuôn mặt thành định dạng mà Cloudinary có thể nhận
        _, buffer = cv2.imencode('.jpg', face)
        face_data = BytesIO(buffer)
         # Upload ảnh lên Cloudinary
        upload_result = cloudinary.uploader.upload(
            face_data,
            resource_type="image",  # Tài nguyên là ảnh
            folder="load/image",  # Thư mục 'load/image'
        )
        print(f"Ảnh đã được upload lên Cloudinary: {upload_result['secure_url']}")


    return frame


@app.route('/', methods= ['GET'])
def home():
    return "Hello Ae",200
@app.route('/capture_face', methods=['POST'])
def capture_face():
    try:
        # Kiểm tra xem có file ảnh hay không
        if 'image' not in request.files:
            return jsonify({"message": "No image file found", "status": 400}), 400

        file = request.files['image']

        # Kiểm tra file có rỗng không
        if file.filename == '':
            return jsonify({"message": "Empty file", "status": 400}), 400

        # Đọc ảnh từ file
        frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Kiểm tra nếu ảnh không đọc được
        if frame is None:
            return jsonify({"message": "Invalid image file", "status": 400}), 400

        # Xử lý và upload ảnh khuôn mặt
        uploaded_urls = process_and_upload_face(frame)

        if not uploaded_urls:
            return jsonify({"message": "No face detected", "status": 400}), 400

        return jsonify({"message": "Processed and uploaded face images.", "status": 200, "uploaded_images": uploaded_urls}), 200

    except Exception as e:
        return jsonify({"message": "Internal server error", "error": str(e), "status": 500}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=port)
