import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import cv2
import cloudinary
import cloudinary.uploader
import cloudinary.api
from flask import Flask, request, jsonify
import numpy as np
from io import BytesIO
import requests

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
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Hàm xử lý ảnh và lưu vào Cloudinary
def process_and_upload_face(frame):

    # Chuyển ảnh sang màu xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong ảnh
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return {"status": 400, "message": "No face detected", "uploaded_images": []}  # Không phát hiện khuôn mặt

    for (x, y, w, h) in faces:
        # Cắt khuôn mặt
        face = frame[y:y+h, x:x+w]
        
        # Vẽ hình chữ nhật quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Chuyển ảnh khuôn mặt thành định dạng mà Cloudinary có thể nhận
        _, buffer = cv2.imencode('.jpg', face)
        face_data = BytesIO(buffer)
        
        # Upload ảnh lên Cloudinary
        cloudinary.uploader.upload(
            face_data,
            resource_type="image",
            folder="load/image",
        )
        return "test"

def process(frame):

    # Chuyển ảnh sang màu xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong ảnh
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return {"status": 400, "message": "No face detected", "uploaded_images": []}  # Không phát hiện khuôn mặt

    for (x, y, w, h) in faces:
        # Cắt khuôn mặt
        face = frame[y:y+h, x:x+w]
        
        # Vẽ hình chữ nhật quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Chuyển ảnh khuôn mặt thành định dạng mà Cloudinary có thể nhận
        _, buffer = cv2.imencode('.jpg', face)
        face_data = BytesIO(buffer)
    return face_data
# Hàm phát hiện và trích xuất khuôn mặt
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None, None  # Không tìm thấy khuôn mặt

    x, y, w, h = faces[0]  # Chọn khuôn mặt đầu tiên
    face_img = gray[y:y+h, x:x+w]
    return face_img, (x, y, w, h)


# Tải ảnh từ Cloudinary và gán ID
def load_faces_from_cloudinary(image_names):
    face_samples = []
    ids = []

    for idx, image_name in enumerate(image_names):
        image_url = f"https://res.cloudinary.com/dphfcojlc/image/upload/{image_name}.jpg"
        response = requests.get(image_url)

        if response.status_code == 200:
            image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            face, _ = detect_face(image)

            if face is not None:
                face_samples.append(face)
                ids.append(idx)  # Gán mỗi ảnh một ID

    return face_samples, np.array(ids)


@app.route('/compare_face', methods=['POST'])
def compare_face():
    try:
        if 'image' not in request.files:
            return jsonify({"message": "No image file found", "status": 400}), 400

        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        test_face, _ = detect_face(image)

        if test_face is None:
            return jsonify({"message": "No face detected in uploaded image", "status": 400}), 400

        image_names = request.form.getlist('image_names')
        if not image_names:
            return jsonify({"message": "No image names provided", "status": 400}), 400

        # Tải khuôn mặt từ Cloudinary và huấn luyện mô hình
        faces, ids = load_faces_from_cloudinary(image_names)

        if len(faces) == 0:
            return jsonify({"message": "No valid faces found in database", "status": 400}), 400

        recognizer.train(faces, ids)  # Huấn luyện mô hình

        # So sánh với ảnh mới
        label, confidence = recognizer.predict(test_face)

        if confidence < 50:  # Ngưỡng nhận diện (càng nhỏ càng chính xác)
            matched_image = image_names[label]
            return jsonify({"message": "Face match found", "matched_image": matched_image, "confidence": confidence, "status": 200}), 200
        else:
            return jsonify({"message": "No match found", "status": 404}), 404

    except Exception as e:
        return jsonify({"message": "Internal server error", "error": str(e), "status": 500}), 500


@app.route('/', methods= ['GET'])
def home():
    return "Hello Ae",200

@app.route('/capture_face', methods=['POST'])
def capture_face():
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


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=port)
