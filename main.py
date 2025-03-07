import cv2
import requests
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import os

port = int(os.environ.get("PORT", 443))

app = Flask(__name__)

# Đường dẫn đến ảnh từ Cloudinary
CLOUDINARY_URL = "https://res.cloudinary.com/dphfcojlc/image/upload/{image_name}.jpg"

# Tải ảnh từ Cloudinary & chuyển thành Base64
def fetch_and_convert_to_base64(image_name):
    url = CLOUDINARY_URL.format(image_name=image_name)
    response = requests.get(url)

    if response.status_code == 200:
        image_bytes = response.content
        return base64.b64encode(image_bytes).decode("utf-8")  # Trả về Base64 string
    return None

# Chuyển Base64 -> NumPy array
def base64_to_numpy(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return np.array(image)

# Resize ảnh về kích thước chuẩn
def resize_image(image_np, target_size=(160, 160)):
    return cv2.resize(image_np, target_size, interpolation=cv2.INTER_AREA)

# Phát hiện khuôn mặt sử dụng OpenCV
def detect_faces(image_np):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Kiểm tra khuôn mặt có trong danh sách ảnh cloud hay không
def match_face_with_cloud_images(user_image_base64, cloud_image_names):
    user_image_np = base64_to_numpy(user_image_base64)
    user_image_np = resize_image(user_image_np)
    user_faces = detect_faces(user_image_np)

    if len(user_faces) == 0:
        return False  # Không tìm thấy khuôn mặt

    for image_name in cloud_image_names:
        cloud_image_base64 = fetch_and_convert_to_base64(image_name)
        if cloud_image_base64 is None:
            continue

        cloud_image_np = base64_to_numpy(cloud_image_base64)
        cloud_image_np = resize_image(cloud_image_np)
        cloud_faces = detect_faces(cloud_image_np)

        if len(cloud_faces) > 0:
            return True  # Nếu có khuôn mặt trong ảnh cloud, coi như hợp lệ

    return False

@app.route('/', methods=['GET'])
def home():
    return "Hello, FaceScan is running!", 200
@app.route('/test', methods=['GET'])
def test():
    haar_cascades_directory = cv2.data.haarcascades
    return jsonify({"Haar cascades directory": haar_cascades_directory}), 200  # In ra đường dẫn thư mục chứa tệp Haar cascades

@app.route('/verify', methods=['POST'])
def verify_face():
    try:
        data = request.json
        user_image_base64 = data['image']  # Ảnh nhận từ Flutter (Base64)
        cloud_image_names = data['image_names']  # Danh sách ảnh cloud

        if match_face_with_cloud_images(user_image_base64, cloud_image_names):
            return jsonify({"status": "success", "message": "Face verified"})

        return jsonify({"status": "failure", "message": "Face not recognized"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)
