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
from deepface import DeepFace

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

# Hàm nhận diện khuôn mặt
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    face_images = []
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_images.append(face)

    return face_images


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

    return {"status": 200, "message": "Processed and uploaded face images.", "uploaded_images": uploaded_urls}

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
        




@app.route('/', methods= ['GET'])
def home():
    return "Hello Ae",200
@app.route('/compare_face', methods=['POST'])
def compare_face():
    try:
        # Kiểm tra xem có file ảnh không
        if 'image' not in request.files:
            return jsonify({"message": "No image file found", "status": 400}), 400

        # Nhận file ảnh từ request
        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image_process = process(image)

        # Kiểm tra xem có danh sách tên ảnh không
        image_names = request.form.getlist('image_names')
        if not image_names:
            return jsonify({"message": "No image names provided", "status": 400}), 400

        # Lưu ảnh tạm thời để so sánh
        temp_image_path = "temp_uploaded.jpg"
        cv2.imwrite(temp_image_path, image_process)

        matched_images = []

        # So sánh với từng ảnh trong danh sách Cloudinary
        for image_name in image_names:
            image_url = f"https://res.cloudinary.com/dphfcojlc/image/upload/{image_name}.jpg"

            # Tải ảnh từ Cloudinary
            response = requests.get(image_url)
            if response.status_code != 200:
                continue

            # Đọc ảnh tải về từ Cloudinary
            cloud_image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            temp_cloud_path = f"temp_cloud_{image_name}.jpg"
            cv2.imwrite(temp_cloud_path, cloud_image)

            # So sánh khuôn mặt bằng DeepFace
            try:
                result = DeepFace.verify(temp_image_path, temp_cloud_path, model_name="Facenet")
                if result["verified"]:
                    matched_images.append(image_name)
            except Exception as e:
                print(f"Error comparing {image_name}: {e}")

        # Xóa ảnh tạm thời
        os.remove(temp_image_path)

        if matched_images:
            return jsonify({"message": "Face match found", "matched_images": matched_images, "status": 200}), 200
        else:
            return jsonify({"message": "No match found", "status": 404}), 404

    except Exception as e:
        return jsonify({"message": "Internal server error", "error": str(e), "status": 500}), 500

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


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=port)
